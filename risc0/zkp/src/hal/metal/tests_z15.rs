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

//! Z15 (segment tail and host round trips): interleaved A/B microbenchmarks (`--ignored`, run
//! alone) and the bit-exactness tests of the kept changes. See docs/metal-port/notes/Z15.md.

use std::time::Instant;

use rand::{rngs::SmallRng, Rng, SeedableRng};
use risc0_core::field::{
    baby_bear::{BabyBearElem, BabyBearExtElem},
    Elem as _,
};

use super::{
    as_bytes,
    tests_z9::{env_or, random_vec},
    BufferImpl, MetalHalPoseidon2,
};
use crate::hal::{Buffer as _, Hal};

type Fp = BabyBearElem;

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    (s[s.len() / 2], s[0], s[s.len() - 1])
}

fn load_avg() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// LEADS #31, second probe: raw command buffers (one tiny or ~0.3 ms dispatch each) completed by
/// `waitUntilCompleted`, by polling `status`, or by polling a shared event the command buffer
/// signals after the dispatch. Prints the round trip per variant.
#[test]
#[ignore]
fn z15_sync_latency_raw() {
    crate::hal::metal::require_gpu_run();
    use metal::{MTLCommandBufferStatus, MTLSize};
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let iters: usize = env_or("Z15_ITERS", 200);
    let pairs: usize = env_or("Z15_PAIRS", 9);
    let queue = hal.device.new_command_queue();
    let event = hal.device.new_shared_event();
    let pso = hal.pipeline("eltwise_copy_fp").to_owned();
    let mut ev = 0u64;
    println!("load {}", load_avg());
    for (what, n) in [("tiny", 1024u64), ("~0.3 ms", 1 << 24)] {
        let src = hal.alloc_elem("s", n as usize);
        let dst = hal.alloc_elem("d", n as usize);
        hal.eltwise_zeroize_elem(&src);
        hal.sync();
        let names = ["block", "status-poll", "event-poll"];
        let mut res: Vec<Vec<f64>> = vec![Vec::new(); 3];
        let iters = if n > 1024 { iters / 4 } else { iters };
        for p in 0..pairs {
            for k0 in 0..3 {
                let k = if p % 2 == 0 { k0 } else { 2 - k0 };
                let t0 = Instant::now();
                for _ in 0..iters {
                    let cmd = queue.new_command_buffer();
                    let enc = cmd.new_compute_command_encoder();
                    enc.set_compute_pipeline_state(&pso);
                    enc.set_buffer(0, Some(dst.as_buf()), 0);
                    enc.set_buffer(1, Some(src.as_buf()), 0);
                    enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(256, 1, 1));
                    enc.end_encoding();
                    ev += 1;
                    if k == 2 {
                        cmd.encode_signal_event(&event, ev);
                    }
                    cmd.commit();
                    match k {
                        0 => cmd.wait_until_completed(),
                        1 => {
                            while !matches!(
                                cmd.status(),
                                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
                            ) {
                                std::hint::spin_loop();
                            }
                        }
                        _ => {
                            while event.signaled_value() < ev {
                                std::hint::spin_loop();
                            }
                        }
                    }
                }
                res[k].push(t0.elapsed().as_secs_f64() * 1e6 / iters as f64);
            }
        }
        for k in 0..3 {
            let (m, lo, hi) = stats(&res[k]);
            println!(
                "{what:8} {:12}: {m:8.1} us per round trip [{lo:.1}, {hi:.1}]",
                names[k]
            );
        }
    }
    println!("load {}", load_avg());
}

/// The rv32im shape of the FRI mix (accum, code, data, check): combo ids per input, as
/// `taps.group_regs(id).combo_id()` gives them for rv32im's TAPSET, plus 16 check inputs.
pub(crate) fn rv32im_mix_ids() -> Vec<Vec<u32>> {
    let runs = |r: &[(u32, usize)]| -> Vec<u32> {
        r.iter()
            .flat_map(|&(c, n)| std::iter::repeat_n(c, n))
            .collect()
    };
    vec![
        runs(&[(1, 12), (0, 87), (1, 4)]),
        runs(&[(0, 1)]),
        runs(&[
            (1, 1),
            (0, 13),
            (1, 4),
            (0, 11),
            (1, 6),
            (0, 1),
            (2, 64),
            (3, 32),
            (0, 79),
        ]),
        vec![4; 16],
    ]
}

fn host_mix(
    mix: BabyBearExtElem,
    inputs: &[(Vec<Fp>, Vec<u32>)],
    count: usize,
    n_combos: usize,
) -> Vec<BabyBearExtElem> {
    let mut out = vec![BabyBearExtElem::ZERO; count * n_combos];
    let mut cur = BabyBearExtElem::ONE;
    for (vals, ids) in inputs {
        for (i, &c) in ids.iter().enumerate() {
            for row in 0..count {
                out[c as usize * count + row] += cur * vals[i * count + row];
            }
            cur *= mix;
        }
    }
    out
}

/// Z15 fused mix: bit-exact against a host reference and against the pre-Z15 path (zero fill +
/// per-group mix) on the rv32im shape and on edge shapes (empty combos, one input, odd runs,
/// unsorted ids, row counts 4 and 4 * 2^k + 4, the old-path fallback for a count that is not a
/// multiple of 4). The output buffer starts with garbage.
#[test]
fn mix_fused_matches_reference() {
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let mut rng = SmallRng::seed_from_u64(0x215);
    let mut shapes: Vec<(Vec<Vec<u32>>, usize, usize)> = vec![
        (rv32im_mix_ids(), 1024, 5),
        (rv32im_mix_ids(), 4, 5),
        (vec![vec![0]], 8, 1),
        (vec![vec![2, 0, 2, 1, 0]], 260, 4), // combo 3 empty, ids unsorted
        (vec![vec![1; 7], vec![0; 3], vec![1, 0], vec![1]], 36, 2),
        (vec![vec![0, 1], vec![1, 0]], 6, 2), // count % 4 != 0: old path
    ];
    for _ in 0..4 {
        let groups = rng.random_range(1..=4);
        let n_combos = rng.random_range(1..=6);
        let ids = (0..groups)
            .map(|_| {
                (0..rng.random_range(1..40))
                    .map(|_| rng.random_range(0..n_combos as u32))
                    .collect()
            })
            .collect();
        shapes.push((ids, 4 * rng.random_range(1..300), n_combos));
    }
    for (ids, count, n_combos) in shapes {
        let mix = BabyBearExtElem::random(&mut rng);
        let inputs: Vec<(Vec<Fp>, Vec<u32>)> = ids
            .iter()
            .map(|ids| (random_vec(ids.len() * count), ids.clone()))
            .collect();
        let want = host_mix(mix, &inputs, count, n_combos);
        let bufs: Vec<_> = inputs
            .iter()
            .map(|(v, _)| hal.copy_from_elem("in", v))
            .collect();
        let args: Vec<(&BufferImpl<Fp>, &[u32])> = bufs
            .iter()
            .zip(inputs.iter())
            .map(|(b, (_, ids))| (b, ids.as_slice()))
            .collect();
        for force_old in [false, true] {
            let garbage: Vec<BabyBearExtElem> = (0..count * n_combos)
                .map(|_| BabyBearExtElem::random(&mut rng))
                .collect();
            let out = hal.copy_from_extelem("out", &garbage);
            let fused =
                super::mix::mix_poly_coeffs_fused(&hal, &out, &mix, &args, count, force_old);
            assert_eq!(
                fused,
                !force_old && count.is_multiple_of(4),
                "count {count}"
            );
            assert!(
                out.to_vec() == want,
                "fused {fused}: {} groups, count {count}, {n_combos} combos",
                ids.len()
            );
        }
    }
}

/// The fused mix on a plain `Hal` round trip: through `has_mix_poly_coeffs_fused` /
/// `mix_poly_coeffs_fused`, with the knob off (`RISC0_METAL_MIX_FUSED=0`) and on.
#[test]
fn mix_fused_knob() {
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let mut rng = SmallRng::seed_from_u64(7);
    let ids = rv32im_mix_ids();
    let count = 64;
    let mix = BabyBearExtElem::random(&mut rng);
    let inputs: Vec<(Vec<Fp>, Vec<u32>)> = ids
        .iter()
        .map(|ids| (random_vec(ids.len() * count), ids.clone()))
        .collect();
    let want = host_mix(mix, &inputs, count, 5);
    let bufs: Vec<_> = inputs
        .iter()
        .map(|(v, _)| hal.copy_from_elem("in", v))
        .collect();
    let args: Vec<(&BufferImpl<Fp>, &[u32])> = bufs
        .iter()
        .zip(inputs.iter())
        .map(|(b, (_, ids))| (b, ids.as_slice()))
        .collect();
    let saved = std::env::var_os(super::mix::MIX_FUSED_ENV);
    for v in ["0", "1"] {
        std::env::set_var(super::mix::MIX_FUSED_ENV, v);
        assert_eq!(hal.has_mix_poly_coeffs_fused(), v == "1");
        let out = hal.alloc_extelem("out", count * 5);
        hal.mix_poly_coeffs_fused(&out, &mix, &args, count);
        assert!(out.to_vec() == want, "knob {v}");
    }
    match saved {
        Some(v) => std::env::set_var(super::mix::MIX_FUSED_ENV, v),
        None => std::env::remove_var(super::mix::MIX_FUSED_ENV),
    }
}

/// LEADS #22 go/no-go: the rv32im FRI mix at `Z15_PO2` (default 20): zero fill + 4 per-group
/// mixes (pre-Z15) against the fused kernel, rotating, encode + sync on the wall clock.
#[test]
#[ignore]
fn z15_mix_bench() {
    crate::hal::metal::require_gpu_run();
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let po2: usize = env_or("Z15_PO2", 20);
    let pairs: usize = env_or("Z15_PAIRS", 15);
    let count = 1usize << po2;
    let ids = rv32im_mix_ids();
    let mut rng = SmallRng::seed_from_u64(1);
    let mix = BabyBearExtElem::random(&mut rng);
    let bufs: Vec<_> = ids
        .iter()
        .map(|ids| hal.copy_from_elem("in", &random_vec(ids.len() * count)))
        .collect();
    let args: Vec<(&BufferImpl<Fp>, &[u32])> = bufs
        .iter()
        .zip(ids.iter())
        .map(|(b, ids)| (b, ids.as_slice()))
        .collect();
    let variants: Vec<(&str, usize)> = vec![("old", 4), ("mix_poly_coeffs_fused", 4)];
    let nv = variants.len();
    let out: Vec<_> = (0..nv)
        .map(|_| hal.alloc_extelem("out", count * 5))
        .collect();
    hal.sync();
    println!("load {}", load_avg());
    let mut t = vec![Vec::new(); nv];
    for p in 0..pairs {
        for k0 in 0..nv {
            let k = (k0 + p) % nv;
            let t0 = Instant::now();
            super::mix::mix_poly_coeffs_fused_with(
                &hal,
                &out[k],
                &mix,
                &args,
                count,
                k == 0,
                variants[k],
            );
            hal.sync();
            t[k].push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    let want = out[0].to_vec();
    for k in 1..nv {
        assert!(out[k].to_vec() == want, "{}", variants[k].0);
    }
    for k in 0..nv {
        let (m, lo, hi) = stats(&t[k]);
        let d: Vec<f64> = (0..pairs).map(|i| t[k][i] - t[0][i]).collect();
        let (dm, dlo, dhi) = stats(&d);
        println!(
            "po2 {po2} {:30}: {m:7.3} ms [{lo:.3}, {hi:.3}]  vs old {dm:+.3} [{dlo:+.3}, {dhi:+.3}]",
            variants[k].0
        );
    }
    println!("load {}", load_avg());
}

/// LEADS #18 go/no-go: eval_u of the rv32im data group at `Z15_PO2` (default 20): 211 polys,
/// 670 evaluation points (taps per register as rv32im: 104 x 1, 11 x 2, 32 x 5, 64 x 6) against
/// (a) the same 670 points all on poly 0 (every coefficient read after the first hits the cache:
/// the floor a multi-point kernel could reach with today's per-point arithmetic) and (b) one
/// point per register (211 points: the read floor of a multi-point kernel).
#[test]
#[ignore]
fn z15_eval_u_bench() {
    crate::hal::metal::require_gpu_run();
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let po2: usize = env_or("Z15_PO2", 20);
    let pairs: usize = env_or("Z15_PAIRS", 15);
    let deg = 1usize << po2;
    let polys = 211;
    let mut which = Vec::new();
    for (k, n) in [(1usize, 104usize), (2, 11), (5, 32), (6, 64)] {
        for _ in 0..n {
            let r = which
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len() as u32;
            for _ in 0..k {
                which.push(r);
            }
        }
    }
    assert_eq!(which.len(), 670);
    let mut rng = SmallRng::seed_from_u64(3);
    let coeffs = hal.copy_from_elem("c", &random_vec(deg * polys));
    let mk = |w: &[u32], rng: &mut SmallRng| {
        let xs: Vec<BabyBearExtElem> = (0..w.len()).map(|_| BabyBearExtElem::random(rng)).collect();
        (
            hal.copy_from_u32("w", w),
            hal.copy_from_extelem("x", &xs),
            hal.alloc_extelem("o", w.len()),
        )
    };
    let per_reg: Vec<u32> = (0..polys as u32).collect();
    let variants = [
        ("670 points (rv32im)", mk(&which, &mut rng)),
        ("670 points on poly 0", mk(&vec![0u32; 670], &mut rng)),
        ("211 points, 1 per reg", mk(&per_reg, &mut rng)),
    ];
    hal.sync();
    println!("load {}", load_avg());
    let nv = variants.len() * 2;
    let mut t = vec![Vec::new(); nv];
    let saved = std::env::var_os(super::eval::EU_LAZY_ENV);
    let mut outs = vec![Vec::new(); nv];
    for p in 0..pairs {
        for k0 in 0..nv {
            let k = (k0 + p) % nv;
            let (w, x, o) = &variants[k / 2].1;
            std::env::set_var(super::eval::EU_LAZY_ENV, if k % 2 == 0 { "0" } else { "1" });
            let t0 = Instant::now();
            hal.batch_evaluate_any(&coeffs, polys, w, x, o);
            hal.sync();
            t[k].push(t0.elapsed().as_secs_f64() * 1e3);
            outs[k] = o.to_vec();
        }
    }
    match saved {
        Some(v) => std::env::set_var(super::eval::EU_LAZY_ENV, v),
        None => std::env::remove_var(super::eval::EU_LAZY_ENV),
    }
    for k in 0..nv {
        let (m, lo, hi) = stats(&t[k]);
        let d: Vec<f64> = (0..pairs).map(|i| t[k][i] - t[k & !1][i]).collect();
        let (dm, dlo, dhi) = stats(&d);
        println!(
            "po2 {po2} {:24} {:8}: {m:7.3} ms [{lo:.3}, {hi:.3}]  vs pairwise {dm:+.3} [{dlo:+.3}, {dhi:+.3}]",
            variants[k / 2].0,
            if k % 2 == 0 { "pairwise" } else { "lazy" }
        );
        assert!(outs[k] == outs[k & !1], "lazy != pairwise");
    }
    println!("load {}", load_avg());
}

/// Z15 lazy eval_u (`eu2_main_lz`): bit-exact against a host Horner evaluation and against the
/// pairwise `eu2_main`, on random data and on worst-case operands (every coefficient and every
/// point component p - 1, the largest canonical value, which maximizes every partial sum the
/// kernel keeps in 64 bits; raw Montgomery words), for S = 1, 2, 4 and degrees 2^12 .. 2^15.
#[test]
fn eu2_lazy_matches_reference() {
    use super::eval::{batch_evaluate_any, EvalPath, EU_LAZY_ENV, EU_MIN_CHUNK};
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let mut rng = SmallRng::seed_from_u64(0x1815);
    // Raw (Montgomery) value p - 1: the largest operand the kernel multiplies.
    let max = Fp::new_raw(risc0_core::field::baby_bear::P - 1);
    let saved = std::env::var_os(EU_LAZY_ENV);
    for po2 in [12usize, 13, 15] {
        let deg = 1usize << po2;
        for worst in [false, true] {
            let polys = 3;
            let coeffs: Vec<Fp> = if worst {
                vec![max; deg * polys]
            } else {
                random_vec(deg * polys)
            };
            let which: Vec<u32> = vec![0, 2, 1, 2, 0];
            let xs: Vec<BabyBearExtElem> = (0..which.len())
                .map(|i| {
                    if worst && i < 3 {
                        BabyBearExtElem::new(max, max, max, max)
                    } else {
                        BabyBearExtElem::random(&mut rng)
                    }
                })
                .collect();
            let want: Vec<BabyBearExtElem> = which
                .iter()
                .zip(xs.iter())
                .map(|(&w, &x)| {
                    let poly = &coeffs[w as usize * deg..(w as usize + 1) * deg];
                    poly.iter()
                        .rev()
                        .fold(BabyBearExtElem::ZERO, |acc, &c| acc * x + c)
                })
                .collect();
            let c = hal.copy_from_elem("c", &coeffs);
            let w = hal.copy_from_u32("w", &which);
            let x = hal.copy_from_extelem("x", &xs);
            for s in [1u32, 2, 4] {
                let chunk = deg as u32 / s;
                if !chunk.is_multiple_of(EU_MIN_CHUNK) {
                    continue;
                }
                for lazy in ["1", "0"] {
                    std::env::set_var(EU_LAZY_ENV, lazy);
                    let o = hal.alloc_extelem("o", which.len());
                    let path = batch_evaluate_any(
                        &hal,
                        &c,
                        polys,
                        &w,
                        &x,
                        &o,
                        Some(EvalPath::Eu2 { s, chunk }),
                    );
                    assert_eq!(path, EvalPath::Eu2 { s, chunk });
                    assert!(
                        o.to_vec() == want,
                        "po2 {po2} worst {worst} S {s} lazy {lazy}"
                    );
                }
            }
        }
    }
    match saved {
        Some(v) => std::env::set_var(EU_LAZY_ENV, v),
        None => std::env::remove_var(EU_LAZY_ENV),
    }
}

/// LEADS #33 probe: `threadGroupSizeIsMultipleOfThreadExecutionWidth` on the PSOs of the Z15
/// kernels (dispatched with whole 256-thread threadgroups only, so the flag's contract holds):
/// `eu2_main_lz` on the rv32im data shape and `mix_poly_coeffs_fused` on the rv32im mix, each
/// flag off / on alternating. Prints medians and per-pair differences.
#[test]
#[ignore]
fn z15_pso_flag_bench() {
    crate::hal::metal::require_gpu_run();
    use metal::{ComputePipelineDescriptor, MTLSize};

    use super::{Grid, KernelArg};
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let pairs: usize = env_or("Z15_PAIRS", 21);
    let lib = &super::zkp_kernels(&hal.device).library;
    let pso = |name: &str, flag: bool| {
        let f = lib.get_function(name, None).unwrap();
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&f));
        desc.set_thread_group_size_is_multiple_of_thread_execution_width(flag);
        hal.device.new_compute_pipeline_state(&desc).unwrap()
    };
    // eu2 on the data shape: deg 2^20, 211 polys, 670 points, S = 2.
    let deg = 1usize << 20;
    let polys = 211;
    let n = 670u64;
    let s = 2u32;
    let coeffs = hal.copy_from_elem("c", &random_vec(deg * polys));
    let which: Vec<u32> = (0..n as u32).map(|i| i * polys as u32 / n as u32).collect();
    let mut rng = SmallRng::seed_from_u64(9);
    let xs: Vec<BabyBearExtElem> = (0..n).map(|_| BabyBearExtElem::random(&mut rng)).collect();
    let (w, x) = (
        hal.copy_from_u32("w", &which),
        hal.copy_from_extelem("x", &xs),
    );
    let pw = hal.alloc_extelem("pw", n as usize * 256);
    let ps = hal.alloc_extelem("ps", n as usize * s as usize);
    let part = hal.alloc_extelem("part", n as usize * s as usize);
    let params: [u32; 4] = [deg as u32, deg as u32 / s, s, 0];
    hal.dispatch(
        hal.pipeline("eu2_pows"),
        &[
            pw.as_arg(),
            ps.as_arg(),
            x.as_arg(),
            KernelArg::Bytes(as_bytes(&params)),
        ],
        Grid::Groups {
            groups: MTLSize::new(n, 1, 1),
            threads: MTLSize::new(256, 1, 1),
        },
        &[],
    );
    hal.sync();
    let eu = [pso("eu2_main_lz", false), pso("eu2_main_lz", true)];
    let eu_run = |k: usize| {
        hal.dispatch(
            &eu[k],
            &[
                part.as_arg(),
                coeffs.as_arg(),
                w.as_arg(),
                x.as_arg(),
                pw.as_arg(),
                ps.as_arg(),
                KernelArg::Bytes(as_bytes(&params)),
            ],
            Grid::Groups {
                groups: MTLSize::new(s as u64, n, 1),
                threads: MTLSize::new(256, 1, 1),
            },
            &[],
        );
        hal.sync();
        part.to_vec()
    };
    // The fused mix, rv32im shape, 2^20 rows (2^18 threads: whole threadgroups).
    let count = 1usize << 20;
    let ids = rv32im_mix_ids();
    let bufs: Vec<_> = ids
        .iter()
        .map(|i| hal.copy_from_elem("in", &random_vec(i.len() * count)))
        .collect();
    let sizes: Vec<usize> = ids.iter().map(Vec::len).collect();
    let flat: Vec<u32> = ids.concat();
    let mix = BabyBearExtElem::random(&mut rng);
    let (pows, entries, begin) = super::mix::fused_tables(&mix, &sizes, &flat, 5).unwrap();
    let (pows, entries, begin) = (
        hal.copy_from_slice("p", &pows),
        hal.copy_from_slice("e", &entries),
        hal.copy_from_slice("b", &begin),
    );
    let out = hal.alloc_extelem("out", count * 5);
    let mparams: [u32; 4] = [count as u32, 5, 0, 0];
    let mx = [
        pso("mix_poly_coeffs_fused", false),
        pso("mix_poly_coeffs_fused", true),
    ];
    let mix_run = |k: usize| {
        hal.dispatch(
            &mx[k],
            &[
                out.as_arg(),
                bufs[0].as_arg(),
                bufs[1].as_arg(),
                bufs[2].as_arg(),
                bufs[3].as_arg(),
                pows.as_arg(),
                entries.as_arg(),
                begin.as_arg(),
                KernelArg::Bytes(as_bytes(&mparams)),
            ],
            Grid::Groups {
                groups: MTLSize::new((count / 4 / 256) as u64, 1, 1),
                threads: MTLSize::new(256, 1, 1),
            },
            &[],
        );
        hal.sync();
        out.to_vec()
    };
    println!("load {}", load_avg());
    for (name, run) in [
        (
            "eu2_main_lz",
            &eu_run as &dyn Fn(usize) -> Vec<BabyBearExtElem>,
        ),
        ("mix_poly_coeffs_fused", &mix_run),
    ] {
        let want = run(0);
        assert!(run(1) == want);
        let mut t = [Vec::new(), Vec::new()];
        for p in 0..pairs {
            for k0 in 0..2 {
                let k = if p % 2 == 0 { k0 } else { 1 - k0 };
                let t0 = Instant::now();
                run(k);
                t[k].push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        let d: Vec<f64> = (0..pairs).map(|i| t[1][i] - t[0][i]).collect();
        let (a, alo, ahi) = stats(&t[0]);
        let (b, blo, bhi) = stats(&t[1]);
        let (dm, dlo, dhi) = stats(&d);
        let neg = d.iter().filter(|&&x| x < 0.0).count();
        println!(
            "{name:22}: flag off {a:.3} [{alo:.3}, {ahi:.3}] on {b:.3} [{blo:.3}, {bhi:.3}] ms; \
             on - off {dm:+.3} [{dlo:+.3}, {dhi:+.3}], {neg}/{pairs} negative"
        );
    }
    println!("load {}", load_avg());
}

/// Z15 follow-up: the eu2 split factor S for the lazy kernel on the rv32im data shape (670
/// points, deg 2^20): eval_plan's S (2) against 1, 4 and 8, rotating.
#[test]
#[ignore]
fn z15_eu_s_bench() {
    crate::hal::metal::require_gpu_run();
    use super::eval::{batch_evaluate_any, EvalPath};
    let _lock = super::gpu_test_lock();
    let hal = MetalHalPoseidon2::new();
    let pairs: usize = env_or("Z15_PAIRS", 15);
    let deg = 1usize << 20;
    let polys = 211;
    let n = 670usize;
    let mut rng = SmallRng::seed_from_u64(5);
    let coeffs = hal.copy_from_elem("c", &random_vec(deg * polys));
    let which: Vec<u32> = (0..n as u32).map(|i| i * polys as u32 / n as u32).collect();
    let xs: Vec<BabyBearExtElem> = (0..n).map(|_| BabyBearExtElem::random(&mut rng)).collect();
    let (w, x) = (
        hal.copy_from_u32("w", &which),
        hal.copy_from_extelem("x", &xs),
    );
    let ss = [2u32, 1, 4, 8];
    let mut t = vec![Vec::new(); ss.len()];
    let mut outs = vec![Vec::new(); ss.len()];
    println!("load {}", load_avg());
    for p in 0..pairs {
        for k0 in 0..ss.len() {
            let k = (k0 + p) % ss.len();
            let o = hal.alloc_extelem("o", n);
            let t0 = Instant::now();
            batch_evaluate_any(
                &hal,
                &coeffs,
                polys,
                &w,
                &x,
                &o,
                Some(EvalPath::Eu2 {
                    s: ss[k],
                    chunk: deg as u32 / ss[k],
                }),
            );
            hal.sync();
            t[k].push(t0.elapsed().as_secs_f64() * 1e3);
            outs[k] = o.to_vec();
        }
    }
    for k in 0..ss.len() {
        assert!(outs[k] == outs[0]);
        let (m, lo, hi) = stats(&t[k]);
        let d: Vec<f64> = (0..pairs).map(|i| t[k][i] - t[0][i]).collect();
        let (dm, dlo, dhi) = stats(&d);
        println!(
            "S {:2}: {m:7.3} ms [{lo:.3}, {hi:.3}]  vs S 2 {dm:+.3} [{dlo:+.3}, {dhi:+.3}]",
            ss[k]
        );
    }
    println!("load {}", load_avg());
}
