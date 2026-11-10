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

use std::sync::{atomic::Ordering, Arc};

use risc0_binfmt::MemoryImage;
use risc0_circuit_rv32im_sys::{
    RawMemoryTransaction, RawPreflightCycle, METAL_KERNEL_NAMES, METAL_LIB,
};
use risc0_core::field::{baby_bear::BabyBear, Elem as _, ExtElem as _};
use risc0_zkp::{
    adapter::CircuitInfo as _,
    core::hash::sha::Sha256HashSuite,
    hal::{cpu::CpuHal, metal::MetalHalSha256, Buffer as _, CircuitHal, Hal},
    INV_RATE,
};
use test_log::test;

use super::*;
use crate::{
    execute::{testutil, DEFAULT_SEGMENT_LIMIT_PO2},
    prove::hal::cpu::CpuCircuitHal,
    zirgen::{
        circuit::{Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE, REGISTER_GROUP_DATA},
        info::NUM_POLY_MIX_POWERS,
        taps::TAPSET,
        CircuitImpl,
    },
    MAX_INSN_CYCLES,
};

#[test]
fn missing_kernels_error_only_when_phase_required() {
    let hal = MetalHalSha256::new();
    let expected: &[(Phase, &[&str])] = &[
        (Phase::Witgen, &["injectCycleCols"]),
        (Phase::Accum, &["r7_not_a_kernel", "finalizeZeroizeAccum"]),
        (Phase::EvalCheck, &[]),
    ];
    let k = Rv32imKernels::load_from(&hal, METAL_LIB, METAL_KERNEL_NAMES, expected).unwrap();
    k.require(Phase::Witgen).unwrap();
    k.require(Phase::EvalCheck).unwrap();
    let err = format!("{:#}", k.require(Phase::Accum).unwrap_err());
    assert!(
        err.contains("Accum") && err.contains("r7_not_a_kernel"),
        "{err}"
    );
    assert!(!err.contains("finalizeZeroizeAccum"), "{err}");
    // Every present listed/expected kernel has a PSO.
    assert!(k.kernels.get("injectCycleCols").is_some());
    assert!(k.kernels.get("finalizeZeroizeAccum").is_some());
}

#[test]
fn listed_kernel_missing_fails_load() {
    let hal = MetalHalSha256::new();
    let err = Rv32imKernels::load_from(&hal, METAL_LIB, &["r7_not_a_kernel"], &[])
        .err()
        .expect("a name listed by rv32im-sys must exist");
    assert!(format!("{err:#}").contains("r7_not_a_kernel"));
}

#[test]
fn real_metallib_loads() {
    let hal = Arc::new(MetalHalPoseidon2::new());
    let circuit_hal = MetalCircuitHalPoseidon2::new(hal).unwrap();
    for name in METAL_KERNEL_NAMES {
        let _ = circuit_hal.ctx().pipeline(name);
    }
    // Each phase either has all its kernels or reports which are missing.
    for &(phase, names) in EXPECTED_KERNELS {
        if let Err(e) = circuit_hal.ctx().require(phase) {
            let e = format!("{e:#}");
            assert!(names.iter().any(|n| e.contains(n)), "{e}");
        }
    }
    warmup().unwrap();
}

#[test]
fn scratch_tables_and_preflight_upload() {
    let hal = Arc::new(MetalHalSha256::new());
    let ctx = MetalRv32imCtx::new(hal.clone()).unwrap();
    let (u8t, u16t) = ctx.tables();
    assert!(u8t.size() >= TABLE_U8_CELLS && u16t.size() >= TABLE_U16_CELLS);
    let (u8b, _) = ctx.tables();
    assert_eq!(
        u8b.as_device_ptr(),
        u8t.as_device_ptr(),
        "tables are reused"
    );

    let mut trace = PreflightTrace::default();
    for i in 0..37u32 {
        trace.cycles.push(RawPreflightCycle {
            state: i,
            pc: 4 * i,
            major: (i % 7) as u8,
            minor: (i % 5) as u8,
            machine_mode: 1,
            padding: 0,
            user_cycle: i + 1,
            txn_idx: i + 2,
            paging_idx: i + 3,
            bigint_idx: i + 4,
            diff_count: [i + 5, i + 6],
        });
    }
    for i in 0..11u32 {
        trace.txns.push(RawMemoryTransaction {
            addr: i,
            cycle: i * 2,
            word: i * 3,
            prev_cycle: i * 4,
            prev_word: i * 5,
        });
    }
    trace.bigint_bytes = (0..13u8).collect();
    let (buf, layout) = ctx.upload_preflight(&trace).unwrap();
    assert_eq!(layout, PreflightLayout::new(37, 11, 13));
    assert_eq!(layout.txns_offset % PF_ALIGN, 0);
    assert_eq!(layout.bigint_offset % PF_ALIGN, 0);
    assert!(layout.txns_offset >= 37 * PREFLIGHT_CYCLE_BYTES);
    assert!(buf.size() >= layout.bytes);
    let bytes = buf.to_vec();
    let cyc = &bytes[layout.cycles_offset..][..37 * PREFLIGHT_CYCLE_BYTES];
    // cycle 5: state=5, pc=20, major=5, minor=0, mm=1, pad, user_cycle=6, ... diff_count[1]=11
    let c5 = &cyc[5 * PREFLIGHT_CYCLE_BYTES..][..PREFLIGHT_CYCLE_BYTES];
    assert_eq!(&c5[0..4], &5u32.to_le_bytes());
    assert_eq!(&c5[4..8], &20u32.to_le_bytes());
    assert_eq!(&c5[8..12], &[5, 0, 1, 0]);
    assert_eq!(&c5[32..36], &11u32.to_le_bytes());
    let t10 = &bytes[layout.txns_offset + 10 * MEMORY_TXN_BYTES..][..MEMORY_TXN_BYTES];
    assert_eq!(&t10[16..20], &50u32.to_le_bytes());
    assert_eq!(
        &bytes[layout.bigint_offset..][..13],
        trace.bigint_bytes.as_slice()
    );
    // A smaller trace reuses the buffer.
    trace.cycles.truncate(3);
    let (buf2, _) = ctx.upload_preflight(&trace).unwrap();
    assert_eq!(buf2.as_device_ptr(), buf.as_device_ptr());
}

fn random_vals(n: usize) -> Vec<Val> {
    let mut rng = rand::rng();
    (0..n).map(|_| Val::random(&mut rng)).collect()
}

/// Runs `CircuitHal::eval_check` of `c` on `hal` over copies of `bufs` and returns `check`.
fn run_eval_check<H, C>(
    hal: &H,
    c: &C,
    bufs: [&[Val]; 5],
    poly_mix: ExtVal,
    po2: usize,
    steps: usize,
) -> Vec<Val>
where
    H: Hal<Elem = Val, ExtElem = ExtVal>,
    C: CircuitHal<H>,
{
    let check = hal.alloc_elem("check", ExtVal::EXT_SIZE * steps * INV_RATE);
    let [code, data, accum, mix, out] = bufs.map(|b| hal.copy_from_elem("in", b));
    c.eval_check(
        &check,
        &[&accum, &code, &data],
        &[&mix, &out],
        poly_mix,
        po2,
        steps,
    );
    c.eval_check_dep();
    check.to_vec()
}

/// R10: the Metal eval_check (baseline `eval_check` kernel) is bit-exact with `CpuCircuitHal`'s on
/// random canonical inputs (`Val::random` is uniform in [0, P)), as hal/cuda.rs tests CUDA. Also
/// checks the CPU fallback (`RISC0_METAL_RV32IM_CPU_PHASES=eval`) and the `CircuitHal` entry
/// point, and reports the GPU memory of the phase (PLAN 9.1).
fn eval_check_matches_cpu(po2: usize) {
    let steps = 1 << po2;
    let domain = steps * INV_RATE;
    let code = random_vals(TAPSET.group_size(REGISTER_GROUP_CODE) * domain);
    let data = random_vals(TAPSET.group_size(REGISTER_GROUP_DATA) * domain);
    let accum = random_vals(TAPSET.group_size(REGISTER_GROUP_ACCUM) * domain);
    let mix = random_vals(CircuitImpl::MIX_SIZE);
    let out = random_vals(CircuitImpl::OUTPUT_SIZE);
    let poly_mix = ExtVal::random(&mut rand::rng());
    let bufs = [&code[..], &data, &accum, &mix, &out];

    let t = std::time::Instant::now();
    let cpu_hal: CpuHal<BabyBear> = CpuHal::new(Sha256HashSuite::new_suite());
    let cpu = run_eval_check(&cpu_hal, &CpuCircuitHal, bufs, poly_mix, po2, steps);
    let cpu_time = t.elapsed();

    let hal = Arc::new(MetalHalSha256::new());
    let metal_c = MetalCircuitHal::new(hal.clone()).unwrap();
    let check = hal.alloc_elem("check", ExtVal::EXT_SIZE * domain);
    let [code_b, data_b, accum_b, mix_b, out_b] = bufs.map(|b| hal.copy_from_elem("in", b));
    let groups = [&accum_b, &code_b, &data_b];
    let globals = [&mix_b, &out_b];
    hal.sync();
    hal.reset_peak_allocated_size();
    let before = hal.allocated_size();
    let t = std::time::Instant::now();
    metal_c
        .eval_check_metal(&check, &groups, &globals, poly_mix, po2, steps)
        .unwrap();
    let gpu_time = t.elapsed();
    let peak = hal.peak_allocated_size().max(hal.allocated_size());
    let gpu = check.to_vec();
    let mismatches = cpu.iter().zip(&gpu).filter(|(a, b)| a != b).count();
    assert_eq!(
        mismatches,
        0,
        "po2 {po2}: {mismatches} of {} differ",
        cpu.len()
    );
    assert_eq!(cpu, gpu);
    let inputs: usize = [&code[..], &data, &accum, &mix, &out]
        .iter()
        .map(|b| b.len() * 4)
        .sum();
    println!(
        "R10 eval_check po2 {po2}: domain {domain}, bit-exact {} elements; GPU {gpu_time:?}, \
         CPU HAL {cpu_time:?}; currentAllocatedSize before phase {before} B, peak during phase \
         {peak} B (phase buffers: inputs {inputs} B + check {} B + poly_mix {} B)",
        gpu.len(),
        gpu.len() * 4,
        NUM_POLY_MIX_POWERS * 16,
    );

    // The CPU fallback over the same Metal buffers.
    let check2 = hal.alloc_elem("check", ExtVal::EXT_SIZE * domain);
    metal_c
        .eval_check_cpu(&check2, &groups, &globals, poly_mix, po2, steps)
        .unwrap();
    assert_eq!(cpu, check2.to_vec(), "CPU fallback, po2 {po2}");

    // The CircuitHal entry point (Metal unless RISC0_METAL_RV32IM_CPU_PHASES selects eval).
    if po2 <= 10 {
        let calls = PHASE_CALLS[Phase::EvalCheck as usize].load(Ordering::Relaxed);
        let via_trait = run_eval_check(hal.as_ref(), &metal_c, bufs, poly_mix, po2, steps);
        assert_eq!(cpu, via_trait, "CircuitHal::eval_check, po2 {po2}");
        assert!(PHASE_CALLS[Phase::EvalCheck as usize].load(Ordering::Relaxed) > calls);
    }
}

#[test]
fn eval_check_matches_cpu_po2_4() {
    eval_check_matches_cpu(4);
}

#[test]
fn eval_check_matches_cpu_po2_10() {
    eval_check_matches_cpu(10);
}

#[test]
fn eval_check_matches_cpu_po2_16() {
    eval_check_matches_cpu(16);
}

/// The same poly_mix buffer is reused across calls with different poly_mix values and po2s.
#[test]
fn eval_check_reuses_poly_mix_buffer() {
    let hal = Arc::new(MetalHalSha256::new());
    let metal_c = MetalCircuitHal::new(hal.clone()).unwrap();
    let cpu_hal: CpuHal<BabyBear> = CpuHal::new(Sha256HashSuite::new_suite());
    let mut pm_ptr = None;
    for po2 in [5, 4, 6] {
        let steps = 1 << po2;
        let domain = steps * INV_RATE;
        let code = random_vals(TAPSET.group_size(REGISTER_GROUP_CODE) * domain);
        let data = random_vals(TAPSET.group_size(REGISTER_GROUP_DATA) * domain);
        let accum = random_vals(TAPSET.group_size(REGISTER_GROUP_ACCUM) * domain);
        let mix = random_vals(CircuitImpl::MIX_SIZE);
        let out = random_vals(CircuitImpl::OUTPUT_SIZE);
        let poly_mix = ExtVal::random(&mut rand::rng());
        let bufs = [&code[..], &data, &accum, &mix, &out];
        let cpu = run_eval_check(&cpu_hal, &CpuCircuitHal, bufs, poly_mix, po2, steps);
        // eval_check_metal directly, so the test holds under RISC0_METAL_RV32IM_CPU_PHASES=eval.
        let check = hal.alloc_elem("check", ExtVal::EXT_SIZE * domain);
        let [code_b, data_b, accum_b, mix_b, out_b] = bufs.map(|b| hal.copy_from_elem("in", b));
        metal_c
            .eval_check_metal(
                &check,
                &[&accum_b, &code_b, &data_b],
                &[&mix_b, &out_b],
                poly_mix,
                po2,
                steps,
            )
            .unwrap();
        assert_eq!(cpu, check.to_vec(), "po2 {po2}");
        let ptr = metal_c
            .ctx()
            .scratch()
            .poly_mix
            .as_ref()
            .unwrap()
            .as_device_ptr();
        assert_eq!(*pm_ptr.get_or_insert(ptr), ptr, "poly_mix buffer reused");
    }
}

#[test]
fn cpu_phases_list_parsing() {
    use super::eval_check::phase_listed;
    assert!(phase_listed("eval", "eval"));
    assert!(phase_listed("witgen, Eval ,accum", "eval"));
    assert!(!phase_listed("witgen,accum", "eval"));
    assert!(!phase_listed("", "eval"));
    assert!(!phase_listed("evaluate", "eval"));
}

/// The default rv32im segment prover of a `metal` build is the Metal one: proving a segment
/// goes through every phase of [MetalCircuitHal], and the seal verifies.
#[test]
fn segment_prover_uses_metal_hal() {
    assert!(
        risc0_zkp::hal::metal::enabled("rv32im"),
        "RISC0_METAL / RISC0_METAL_SKIP disable rv32im"
    );
    let before: Vec<usize> = PHASE_CALLS
        .iter()
        .map(|c| c.load(Ordering::Relaxed))
        .collect();
    let image = MemoryImage::new_kernel(testutil::kernel::basic());
    let result = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        MAX_INSN_CYCLES,
        testutil::DEFAULT_SESSION_LIMIT,
        &testutil::NullSyscall,
        None,
    )
    .unwrap();
    let segment = result.segments.first().unwrap();
    let prover = crate::prove::segment_prover().unwrap();
    let seal = prover.prove(segment).unwrap();
    crate::verify(&seal).unwrap();
    for (phase, (c, b)) in [Phase::Witgen, Phase::Accum, Phase::EvalCheck]
        .iter()
        .zip(PHASE_CALLS.iter().zip(before))
    {
        assert!(
            c.load(Ordering::Relaxed) > b,
            "{phase:?} did not run on the Metal circuit HAL"
        );
    }
}
