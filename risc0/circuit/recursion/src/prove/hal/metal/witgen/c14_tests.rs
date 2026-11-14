// Copyright 2026 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! C14 tests: the GPU WOM counting sort (rec_sort.metal) against the host sort on synthetic
//! rows (every scan size class, the flags), and the Parallel witgen with the GPU sort, with it
//! off and with its host fallback forced, against the CPU witgen.

use std::sync::Arc;

use rand::{Rng, SeedableRng};
use risc0_zkp::hal::{metal::MetalHalPoseidon2, Buffer as _};

use super::*;

struct Run {
    hal: Arc<MetalHalPoseidon2>,
    circuit: MetalCircuitHal<risc0_zkp::hal::metal::MetalHashPoseidon2>,
}

impl Run {
    fn new() -> Self {
        let hal = Arc::new(MetalHalPoseidon2::new());
        let circuit = MetalCircuitHal::new(hal.clone()).unwrap();
        Self { hal, circuit }
    }

    fn buf(&self, words: &[u32]) -> BufferImpl<u32> {
        let b = self.hal.alloc_buffer::<u32>("c14", words.len().max(1));
        b.view_mut(|v| v[..words.len()].copy_from_slice(words));
        b.slice(0, words.len().max(1))
    }

    /// (sorted rows, scanned counts, flags) of the GPU sort.
    fn gpu_sort(
        &self,
        rows: &[WomRow],
        counts: &[u32],
        num_woms: usize,
    ) -> (Vec<WomRow>, Vec<u32>, u32) {
        let flat: Vec<u32> = rows.iter().flatten().copied().collect();
        let (wr, wi) = (self.buf(&flat), self.buf(counts));
        let g = self
            .circuit
            .encode_gpu_sort(counts.len(), num_woms, &wr, &wi)
            .expect("gpu sort encodes");
        self.circuit.sync().unwrap();
        let sorted = g.sorted.to_vec();
        let rows: Vec<WomRow> = sorted
            .chunks_exact(WOM_ROW_WORDS)
            .map(|c| c.try_into().unwrap())
            .collect();
        // The inputs are untouched (the host fallback sorts them).
        assert_eq!(wr.to_vec(), flat, "GPU sort modified the exec rows");
        assert_eq!(wi.to_vec(), counts, "GPU sort modified the counts");
        (rows, g.scan.to_vec(), g.sums.get_at(g.flags_at))
    }
}

fn enc(x: u32) -> u32 {
    BabyBearElem::new(x).as_u32_montgomery()
}

/// A consistent WOM: every address < `num_woms` holds one value, rows per cycle 0..=9.
fn consistent_rows(rng: &mut impl Rng, cycles: usize, num_woms: usize) -> (Vec<WomRow>, Vec<u32>) {
    let mut rows = vec![[INVALID; WOM_ROW_WORDS]; cycles * MAX_WOM_ROWS_PER_CYCLE];
    let mut counts = vec![0u32; cycles];
    for c in 0..cycles {
        let n = rng.random_range(0..=9u32);
        counts[c] = n;
        for j in 0..n as usize {
            let a = rng.random_range(0..num_woms as u32);
            let v = a.wrapping_mul(2654435761) % 0x7800_0000;
            rows[c * 9 + j] = [enc(a), enc(v), enc(v ^ 1), enc(3), enc(a ^ 5)];
        }
    }
    (rows, counts)
}

fn host_sort(rows: &[WomRow], counts: &[u32]) -> (Vec<WomRow>, Vec<u32>) {
    let (mut r, mut c) = (rows.to_vec(), counts.to_vec());
    sort_wom_rows(&mut r, &mut c).unwrap();
    (r, c)
}

#[test]
fn gpu_sort_matches_host_sort() {
    let run = Run::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(14);
    // Scan sizes: one partial block, several blocks, and (1.2M addresses) more than 1024
    // block totals, which recScanSums scans in chunks with a carry.
    for (cycles, num_woms) in [
        (1, 4),
        (2, 8),
        (100, 300),
        (1500, 1025),
        (5000, 20000),
        (70000, 200000),
        (300000, 1_200_000),
    ] {
        let (rows, counts) = consistent_rows(&mut rng, cycles, num_woms);
        let (want_rows, want_scan) = host_sort(&rows, &counts);
        let (got_rows, got_scan, flags) = run.gpu_sort(&rows, &counts, num_woms);
        assert_eq!(flags, 0, "{cycles} cycles / {num_woms} addresses: flags");
        assert!(
            got_rows == want_rows,
            "{cycles} cycles / {num_woms} addresses: rows differ"
        );
        assert_eq!(got_scan, want_scan, "{cycles} cycles: scan differs");
    }
    // Scratch reuse: a smaller sort after the largest one.
    let (rows, counts) = consistent_rows(&mut rng, 777, 3000);
    let (want_rows, want_scan) = host_sort(&rows, &counts);
    let (got_rows, got_scan, flags) = run.gpu_sort(&rows, &counts, 3000);
    assert_eq!(flags, 0);
    assert!(got_rows == want_rows);
    assert_eq!(got_scan, want_scan);
}

#[test]
fn gpu_sort_flags() {
    let run = Run::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(41);
    let (rows, counts) = consistent_rows(&mut rng, 4000, 9000);
    let first = (0..counts.len()).find(|&c| counts[c] > 0).unwrap();
    let slot = first * 9;

    // Two different rows at one address: the order is the host's to decide.
    let mut r = rows.clone();
    r[slot][4] = enc(123_456);
    let (_, _, flags) = run.gpu_sort(&r, &counts, 9000);
    assert_eq!(flags & 4, 4, "bucket flag");

    // An address outside the histogram.
    let mut r = rows.clone();
    r[slot][0] = enc(9000);
    let (_, _, flags) = run.gpu_sort(&r, &counts, 9000);
    assert_eq!(flags & 1, 1, "address flag");

    // More than 9 rows in one cycle.
    let mut c = counts.clone();
    c[first] = 10;
    let (_, _, flags) = run.gpu_sort(&rows, &c, 9000);
    assert_eq!(flags & 2, 2, "count flag");

    // And a clean run on the same scratch afterwards.
    let (_, _, flags) = run.gpu_sort(&rows, &counts, 9000);
    assert_eq!(flags, 0);
}

/// The Parallel witgen on the test program at po2 16 with the GPU sort (default), with it off,
/// and with the host fallback forced: each equals the CPU witgen (c6's harness), and the call
/// counts show which sort ran.
#[test]
fn witgen_gpu_sort_paths_match_cpu() {
    use super::c6_tests::witgen_matches_cpu;
    for (label, on, fallback) in [
        ("gpu", true, false),
        ("host", false, false),
        ("gpu+fallback", true, true),
    ] {
        let (sorts, fallbacks) = (
            thread_calls(Call::GpuSort),
            thread_calls(Call::GpuSortFallback),
        );
        with_gpu_sort(on, || {
            if fallback {
                with_gpu_sort_fallback(|| witgen_matches_cpu(16))
            } else {
                witgen_matches_cpu(16)
            }
        });
        let (ds, df) = (
            thread_calls(Call::GpuSort) - sorts,
            thread_calls(Call::GpuSortFallback) - fallbacks,
        );
        eprintln!("C14 witgen {label}: gpu sorts {ds}, fallbacks {df}");
        match (on, fallback) {
            (true, false) => assert!(ds > 0 && df == 0, "{label}"),
            (false, _) => assert!(ds == 0 && df == 0, "{label}"),
            (true, true) => assert!(ds == 0 && df > 0, "{label}"),
        }
    }
}

/// Staged inputs (C14 #12): the trace and host chains in a prepare()-style shared buffer, bound
/// in place, give exactly the CPU witness (with the production chains, with every chain of >= 2
/// cycles, and without chains), and are counted as staged; the caller's trace stays pristine and
/// the staged cycles are not modified (the checks re-run the witgen on them).
fn staged_witgen_matches_cpu(po2: usize) {
    use super::{
        super::staging::MetalStaged,
        c6_tests::{
            assert_same, chains_with_min_len, cpu_witness, preflight_for, raw_trace, MetalRun,
        },
    };
    use crate::prove::witgen::host_chains;
    let (program, pf) = preflight_for(po2);
    let total = 1usize << po2;
    let cpu = cpu_witness(&program, &pf, StepMode::Parallel);
    let pristine = pf.trace.cycles.clone();
    let m = MetalRun::new();
    let small = chains_with_min_len(&program, &pf, 2).expect("chains of >= 2 cycles");
    let prod = host_chains(&program, &pf);
    for (label, chains) in [
        ("no chains", None),
        ("production chains", prod),
        ("chains of >= 2", Some(small)),
    ] {
        let t = &pf.trace;
        let st = MetalStaged::new(&t.wom, &t.cycles, &t.iops, chains.as_ref()).unwrap();
        let mut trace = raw_trace(&program, &pf);
        trace.wom = st.ptr(0);
        trace.cycles = st.ptr(1);
        trace.iops = st.ptr(2);
        let chains = chains.map(|mut hc| {
            hc.pinned = Some(std::array::from_fn(|i| st.ptr::<u32>(3 + i) as usize));
            hc
        });
        let staged_cycles: Vec<u32> =
            unsafe { std::slice::from_raw_parts(st.ptr::<u32>(1), pristine.len() * CYCLE_WORDS) }
                .to_vec();
        let (n_staged, n_chain) = (thread_calls(Call::Staged), thread_calls(Call::Chain));
        for _ in 0..2 {
            let (d, g, _) = m
                .witness(&program, &trace, StepMode::Parallel, chains.as_ref())
                .unwrap();
            assert_same(label, total, &(d, g), &cpu);
        }
        assert_eq!(thread_calls(Call::Staged) - n_staged, 2, "{label}: staged");
        let with_chains = chains.as_ref().is_some_and(|hc| !hc.heads.is_empty());
        assert_eq!(
            thread_calls(Call::Chain) - n_chain,
            if with_chains { 2 } else { 0 },
            "{label}: chains"
        );
        let after: &[u32] =
            unsafe { std::slice::from_raw_parts(st.ptr::<u32>(1), pristine.len() * CYCLE_WORDS) };
        assert_eq!(after, &staged_cycles[..], "{label}: staged cycles modified");
        eprintln!(
            "C14 staged witgen po2 {po2} {label}: {} chain heads",
            chains.as_ref().map_or(0, |hc| hc.heads.len())
        );
    }
    assert_eq!(pf.trace.cycles, pristine, "caller's trace modified");
}

#[test]
fn staged_witgen_matches_cpu_po2_16() {
    super::super::staging::with_stage(true, || staged_witgen_matches_cpu(16));
}

#[test]
fn staged_witgen_matches_cpu_po2_18() {
    super::super::staging::with_stage(true, || staged_witgen_matches_cpu(18));
}

/// With staging off (the default; `RISC0_METAL_REC_STAGE` unset or 0), a staged trace is copied on the host path.
#[test]
fn staged_trace_with_staging_off_takes_host_path() {
    use super::{
        super::staging::{with_stage, MetalStaged},
        c6_tests::{assert_same, cpu_witness, preflight_for, raw_trace, MetalRun},
    };
    let (program, pf) = preflight_for(16);
    let cpu = cpu_witness(&program, &pf, StepMode::Parallel);
    let t = &pf.trace;
    let st = MetalStaged::new(&t.wom, &t.cycles, &t.iops, None).unwrap();
    let mut trace = raw_trace(&program, &pf);
    trace.wom = st.ptr(0);
    trace.cycles = st.ptr(1);
    trace.iops = st.ptr(2);
    let n = thread_calls(Call::Staged);
    let (d, g, _) = with_stage(false, || {
        MetalRun::new()
            .witness(&program, &trace, StepMode::Parallel, None)
            .unwrap()
    });
    assert_same("staging off", 1 << 16, &(d, g), &cpu);
    assert_eq!(thread_calls(Call::Staged), n);
}

/// Unstaged pointers (a plain host trace) take the host path even with staging enabled.
#[test]
fn unstaged_trace_takes_host_path() {
    use super::c6_tests::{preflight_for, raw_trace, MetalRun};
    let (program, pf) = preflight_for(16);
    let trace = raw_trace(&program, &pf);
    let n = thread_calls(Call::Staged);
    MetalRun::new()
        .witness(&program, &trace, StepMode::Parallel, None)
        .unwrap();
    assert_eq!(thread_calls(Call::Staged), n);
}

/// Fault injection on the staged path (C14 #12): the per-element chain ranges are checked by the
/// kernels there (`markHostChains` for the heads, `scatterHostChains` for the chain cycles) instead
/// of on the host (a bad `wom_idx` is flagged by the GPU sort and rejected by its host-sort
/// fallback), and a bad element must fail the witgen with an error, not give a wrong witness.
/// Runs with `RISC0_METAL_DEBUG` on, so the release check also runs on these error paths; the HAL
/// recovers and a good staged trace still matches the CPU.
#[test]
fn staged_chain_faults() {
    use super::{
        super::staging::{with_stage, MetalStaged},
        c6_tests::{
            assert_same, chains_with_min_len, cpu_witness, preflight_for, raw_trace, MetalRun,
        },
    };
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    let (program, pf) = preflight_for(16);
    let m = MetalRun::new();
    m.hal.set_debug(true);
    let chains = || chains_with_min_len(&program, &pf, 2).expect("chains of >= 2 cycles");
    let num_cycles = pf.trace.cycles.len() as u32;
    type Corrupt = fn(&mut crate::prove::witgen::HostChains, u32);
    let cases: [(&str, Corrupt); 5] = [
        ("head past the trace", |hc, _| hc.heads[0] = u32::MAX),
        ("head == numCycles", |hc, n| {
            let last = hc.heads.len() - 1;
            hc.heads[last] = n
        }),
        ("chain cycle past the trace", |hc, _| {
            hc.cycles[0] = u32::MAX
        }),
        ("chain cycle == numCycles", |hc, n| {
            let last = hc.cycles.len() - 1;
            hc.cycles[last] = n
        }),
        // A WOM row count past 9: the GPU sort flags it, and the host-sort fallback rejects it.
        ("wom_idx past 9 rows", |hc, _| hc.wom_idx[0] = 10),
    ];
    let run = |mut hc: crate::prove::witgen::HostChains| {
        let t = &pf.trace;
        let st = MetalStaged::new(&t.wom, &t.cycles, &t.iops, Some(&hc)).unwrap();
        let mut trace = raw_trace(&program, &pf);
        trace.wom = st.ptr(0);
        trace.cycles = st.ptr(1);
        trace.iops = st.ptr(2);
        hc.pinned = Some(std::array::from_fn(|i| st.ptr::<u32>(3 + i) as usize));
        let n = thread_calls(Call::Staged);
        let r = with_stage(true, || {
            m.witness(&program, &trace, StepMode::Parallel, Some(&hc))
        });
        assert_eq!(thread_calls(Call::Staged) - n, 1, "staged path not taken");
        r
    };
    for (label, corrupt) in cases {
        let mut hc = chains();
        corrupt(&mut hc, num_cycles);
        let err = run(hc).expect_err(label);
        let msg = format!("{err:#}");
        assert!(
            msg.contains("fault") || msg.contains("WOM rows (max 9)"),
            "{label}: {msg}"
        );
        eprintln!("C14 staged fault: {label} -> {msg}");
    }
    let cpu = cpu_witness(&program, &pf, StepMode::Parallel);
    let (d, g, _) = run(chains()).unwrap();
    assert_same("staged after faults", 1 << 16, &(d, g), &cpu);
}

/// `RISC0_METAL_DEBUG` covers the staged buffers (C14 #12): the HAL's in-flight tracking records a
/// raw `KernelArg::Buffer` binding, so the witgen's release check would panic on a staged buffer an
/// unsynced batch still reads, and passes once synced; a staged witgen in debug mode passes it.
#[test]
fn debug_check_covers_staged_buffers() {
    use super::{
        super::staging::{lookup, with_stage, MetalStaged},
        c6_tests::{assert_same, cpu_witness, preflight_for, raw_trace, MetalRun},
    };
    let _guard = crate::prove::hal::metal::global_state_test_lock();
    let (program, pf) = preflight_for(16);
    let m = MetalRun::new();
    m.hal.set_debug(true);
    let t = &pf.trace;
    let st = MetalStaged::new(&t.wom, &t.cycles, &t.iops, None).unwrap();
    let words = t.cycles.len() * CYCLE_WORDS;
    let (buf, off) = lookup(st.ptr::<u8>(1), words * 4).expect("staged cycles");
    let dst = m.hal.alloc_buffer::<u32>("c14_dst", words);
    let k = m.circuit.kernels();
    m.hal.dispatch(
        k.pipeline("recCopyWords"),
        &[
            KernelArg::Buffer {
                buffer: &buf,
                offset: off,
            },
            dst.as_arg(),
            KernelArg::Integer(words as u32),
        ],
        Grid::Threads(words as u64),
        &[],
    );
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        m.hal.debug_check_host_access(&buf, "staged", "release")
    }));
    let msg = res.expect_err("an unsynced staged buffer must be caught");
    let msg = msg.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("RISC0_METAL_DEBUG"), "{msg}");
    m.hal.sync();
    m.hal.debug_check_host_access(&buf, "staged", "release");
    let staged_cycles: &[u32] = unsafe { std::slice::from_raw_parts(st.ptr::<u32>(1), words) };
    assert_eq!(dst.to_vec(), staged_cycles);

    let mut trace = raw_trace(&program, &pf);
    trace.wom = st.ptr(0);
    trace.cycles = st.ptr(1);
    trace.iops = st.ptr(2);
    let n = thread_calls(Call::Staged);
    let (d, g, _) = with_stage(true, || {
        m.witness(&program, &trace, StepMode::Parallel, None)
            .unwrap()
    });
    assert_eq!(thread_calls(Call::Staged) - n, 1);
    assert_same(
        "staged, debug",
        1 << 16,
        &(d, g),
        &cpu_witness(&program, &pf, StepMode::Parallel),
    );
}

/// R17-style determinism stress of the GPU sort (ignored; `R17_AUDIT_REPS`, default 30): a
/// join-sized consistent WOM (260K cycles, 564K addresses) sorted repeatedly must equal the host
/// sort every time.
#[test]
#[ignore]
fn c14_gpu_sort_repeat() {
    let reps: usize = std::env::var("R17_AUDIT_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let run = Run::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(1414);
    let (rows, counts) = consistent_rows(&mut rng, 259_612, 564_091);
    let (want_rows, want_scan) = host_sort(&rows, &counts);
    let mut bad = 0;
    for _ in 0..reps {
        let (got_rows, got_scan, flags) = run.gpu_sort(&rows, &counts, 564_091);
        if flags != 0 || got_rows != want_rows || got_scan != want_scan {
            bad += 1;
        }
    }
    println!("R17AUDIT recursion gpu_sort: {bad}/{reps} reps differ from the host sort");
    assert_eq!(bad, 0);
}
