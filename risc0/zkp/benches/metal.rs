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

//! Per-method timings of the Metal HAL (plan 4.1 Z6, 6.4).
//!
//! ```sh
//! source /Users/user/zkminer/env.sh
//! cargo bench -p risc0-zkp --features metal --bench metal
//! RISC0_METAL_BENCH_PO2=20,21 RISC0_METAL_BENCH_COLS=211 cargo bench -p risc0-zkp --features metal --bench metal
//! ```
//!
//! Knobs: `RISC0_METAL_BENCH_PO2` (comma list, default `16,18,20`), `RISC0_METAL_BENCH_COLS`
//! (default 211, the rv32im data group width), `RISC0_METAL_BENCH_ITERS` (default 5),
//! `RISC0_METAL_BENCH_HASH` (`poseidon2` default, or `sha256`), `RISC0_METAL_BENCH_FILTER`
//! (substring of the method name), and the HAL's own knobs (e.g.
//! `RISC0_METAL_MAX_WORKING_SET_BYTES`).
//!
//! Each method is timed as wall clock around one call plus `sync()` (encode + commit + GPU +
//! wait; host-side validation/upload included where the method does it), one warm-up call, then
//! `ITERS` calls; min and median are printed. `GB/s` is (bytes read + bytes written once) / min
//! time, an effective figure (multi-pass kernels touch memory more often). `Mperm/s` counts
//! Poseidon2/SHA permutations. `peak MiB` is the device's highest `currentAllocatedSize` while the
//! method's inputs existed and it ran (inputs included; the pool is trimmed between methods).
//! Timings are indicative: other GPU work on the machine changes them.

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
fn main() {
    eprintln!("the metal bench needs --features metal on macOS");
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
fn main() {
    bench::main();
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod bench {
    use std::time::{Duration, Instant};

    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use risc0_core::field::{
        baby_bear::{BabyBearElem, BabyBearExtElem},
        Elem,
    };
    use risc0_zkp::{
        core::digest::Digest,
        hal::{
            metal::{MetalHal, MetalHalPoseidon2, MetalHalSha256, MetalHash},
            Hal,
        },
        INV_RATE,
    };

    type Fp = BabyBearElem;
    type FpExt = BabyBearExtElem;

    fn env_list(name: &str, default: &str) -> Vec<usize> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect()
    }

    fn env_usize(name: &str, default: usize) -> usize {
        std::env::var(name)
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(default)
    }

    struct Ctx<'a, MH: MetalHash> {
        hal: &'a MetalHal<MH>,
        iters: usize,
        filter: Option<String>,
        po2: usize,
        cols: usize,
        rng: SmallRng,
    }

    impl<MH: MetalHash> Ctx<'_, MH> {
        fn elems(&mut self, n: usize) -> Vec<Fp> {
            (0..n).map(|_| Fp::random(&mut self.rng)).collect()
        }

        fn exts(&mut self, n: usize) -> Vec<FpExt> {
            (0..n).map(|_| FpExt::random(&mut self.rng)).collect()
        }

        fn wants(&self, name: &str) -> bool {
            self.filter.as_deref().is_none_or(|f| name.contains(f))
        }

        /// Time `f` (+ sync) and print one row. `bytes` moved once and `perms` per call.
        fn run(&self, name: &str, bytes: f64, perms: f64, peak_base: u64, mut f: impl FnMut()) {
            let hal = self.hal;
            f();
            hal.sync();
            let mut times: Vec<Duration> = (0..self.iters)
                .map(|_| {
                    let t = Instant::now();
                    f();
                    hal.sync();
                    t.elapsed()
                })
                .collect();
            times.sort();
            let min = times[0].as_secs_f64();
            let med = times[times.len() / 2].as_secs_f64();
            let peak = hal.peak_allocated_size().max(peak_base);
            let gbs = if bytes > 0.0 {
                format!("{:9.1}", bytes / min / 1e9)
            } else {
                format!("{:>9}", "-")
            };
            let mps = if perms > 0.0 {
                format!("{:9.1}", perms / min / 1e6)
            } else {
                format!("{:>9}", "-")
            };
            println!(
                "{:>3} {:>4}  {:<34} {:9.3} {:9.3} {gbs} {mps} {:9}",
                self.po2,
                self.cols,
                name,
                min * 1e3,
                med * 1e3,
                peak >> 20,
            );
        }

        /// Run `body` with a clean pool and a fresh peak.
        fn method(&mut self, name: &str, body: impl FnOnce(&mut Self, &str)) {
            if !self.wants(name) {
                return;
            }
            self.hal.trim_device_memory();
            self.hal.reset_peak_allocated_size();
            body(self, name);
            self.hal.trim_device_memory();
        }
    }

    fn run_shape<MH: MetalHash>(hal: &MetalHal<MH>, po2: usize, cols: usize, iters: usize) {
        let mut cx = Ctx {
            hal,
            iters,
            filter: std::env::var("RISC0_METAL_BENCH_FILTER").ok(),
            po2,
            cols,
            rng: SmallRng::seed_from_u64(po2 as u64 * 1000 + cols as u64),
        };
        let rows = 1usize << po2;
        let n = rows * cols;
        let lde = n * INV_RATE;
        let w = 4.0; // bytes per Fp

        cx.method("sync (empty batch)", |cx, name| {
            cx.run(name, 0.0, 0.0, 0, || cx.hal.sync());
        });
        cx.method("1-thread dispatch + sync", |cx, name| {
            let a = cx.hal.copy_from_elem("a", &[Fp::ONE]);
            let b = cx.hal.alloc_elem("b", 1);
            cx.run(name, 0.0, 0.0, 0, || cx.hal.eltwise_copy_elem(&b, &a));
        });
        cx.method("eltwise_copy_elem", |cx, name| {
            let v = cx.elems(n);
            let a = cx.hal.copy_from_elem("a", &v);
            let b = cx.hal.alloc_elem("b", n);
            cx.run(name, 2.0 * w * n as f64, 0.0, 0, || {
                cx.hal.eltwise_copy_elem(&b, &a)
            });
        });
        cx.method("eltwise_add_elem", |cx, name| {
            let v = cx.elems(n);
            let a = cx.hal.copy_from_elem("a", &v);
            let b = cx.hal.alloc_elem("b", n);
            cx.run(name, 3.0 * w * n as f64, 0.0, 0, || {
                cx.hal.eltwise_add_elem(&b, &a, &a)
            });
        });
        cx.method("eltwise_zeroize_elem", |cx, name| {
            let v = cx.elems(n);
            let a = cx.hal.copy_from_elem("a", &v);
            cx.run(name, 2.0 * w * n as f64, 0.0, 0, || {
                cx.hal.eltwise_zeroize_elem(&a)
            });
        });
        cx.method("alloc_elem_init (fill_u32)", |cx, name| {
            let value = Fp::new(5);
            cx.run(name, w * n as f64, 0.0, 0, || {
                drop(cx.hal.alloc_elem_init("init", n, value))
            });
        });
        cx.method("batch_bit_reverse", |cx, name| {
            let v = cx.elems(lde);
            let a = cx.hal.copy_from_elem("a", &v);
            cx.run(name, 2.0 * w * lde as f64, 0.0, 0, || {
                cx.hal.batch_bit_reverse(&a, cols)
            });
        });
        cx.method("batch_expand_into_evaluate_ntt", |cx, name| {
            let v = cx.elems(n);
            let a = cx.hal.copy_from_elem("a", &v);
            let out = cx.hal.alloc_elem("out", lde);
            cx.run(name, w * (n + lde) as f64, 0.0, 0, || {
                cx.hal.batch_expand_into_evaluate_ntt(&out, &a, cols, 2)
            });
        });
        cx.method("batch_interpolate_ntt", |cx, name| {
            let v = cx.elems(n);
            let a = cx.hal.copy_from_elem("a", &v);
            cx.run(name, 2.0 * w * n as f64, 0.0, 0, || {
                cx.hal.batch_interpolate_ntt(&a, cols)
            });
        });
        cx.method("batch_interpolate_ntt_zk_shift_from", |cx, name| {
            let v = cx.elems(n);
            let a = cx.hal.copy_from_elem("a", &v);
            let out = cx.hal.alloc_elem("out", n);
            cx.run(name, 2.0 * w * n as f64, 0.0, 0, || {
                cx.hal.batch_interpolate_ntt_zk_shift_from(&out, &a, cols)
            });
        });
        cx.method("zk_shift", |cx, name| {
            let v = cx.elems(n);
            let a = cx.hal.copy_from_elem("a", &v);
            cx.run(name, 2.0 * w * n as f64, 0.0, 0, || {
                cx.hal.zk_shift(&a, cols)
            });
        });
        cx.method("commit_coeffs_lde", |cx, name| {
            if !cx.hal.has_commit_coeffs_lde(po2) {
                return;
            }
            let v = cx.elems(n);
            let witness = cx.hal.copy_from_elem("witness", &v);
            let coeffs = cx.hal.alloc_elem("coeffs", n);
            let evaluated = cx.hal.alloc_elem("evaluated", lde);
            cx.run(name, w * (2 * n + lde) as f64, 0.0, 0, || {
                cx.hal
                    .commit_coeffs_lde(&coeffs, &evaluated, &witness, cols)
            });
        });
        cx.method("hash_rows (LDE rows)", |cx, name| {
            let v = cx.elems(lde);
            let matrix = cx.hal.copy_from_elem("matrix", &v);
            let out = cx.hal.alloc_digest("out", rows * INV_RATE);
            let perms = (rows * INV_RATE * cols.div_ceil(16).max(1)) as f64;
            cx.run(name, w * lde as f64, perms, 0, || {
                cx.hal.hash_rows(&out, &matrix)
            });
        });
        cx.method("hash_fold_tree (LDE leaves)", |cx, name| {
            let leaves = rows * INV_RATE;
            let layers = po2 + 2;
            let digests: Vec<Digest> = (0..2 * leaves)
                .map(|_| Digest::from(std::array::from_fn(|_| cx.rng.random::<u32>() / 3)))
                .collect();
            let io = cx.hal.copy_from_digest("io", &digests);
            cx.run(name, 0.0, (leaves - 1) as f64, 0, || {
                cx.hal.hash_fold_tree(&io, layers)
            });
        });
        cx.method("mix_poly_coeffs (input_size=cols)", |cx, name| {
            let ncombos = 5;
            let combos: Vec<u32> = (0..cols)
                .map(|i| (i * ncombos / cols.max(1)) as u32)
                .collect();
            let v = cx.elems(n);
            let input = cx.hal.copy_from_elem("input", &v);
            let out = cx.hal.alloc_extelem_zeroed("out", rows * ncombos);
            let combos = cx.hal.copy_from_u32("combos", &combos);
            let (start, mix) = (FpExt::random(&mut cx.rng), FpExt::random(&mut cx.rng));
            let bytes = w * n as f64 + 2.0 * 16.0 * (rows * ncombos) as f64;
            cx.run(name, bytes, 0.0, 0, || {
                cx.hal
                    .mix_poly_coeffs(&out, &start, &mix, &input, &combos, cols, rows)
            });
        });
        cx.method("batch_evaluate_any (2 evals/poly)", |cx, name| {
            let v = cx.elems(n);
            let coeffs = cx.hal.copy_from_elem("coeffs", &v);
            let evals = 2 * cols;
            let which: Vec<u32> = (0..evals as u32).map(|i| i / 2).collect();
            let xs = cx.exts(evals);
            let which = cx.hal.copy_from_u32("which", &which);
            let xs = cx.hal.copy_from_extelem("xs", &xs);
            let out = cx.hal.alloc_extelem("out", evals);
            cx.run(name, 2.0 * w * n as f64, 0.0, 0, || {
                cx.hal.batch_evaluate_any(&coeffs, cols, &which, &xs, &out)
            });
        });
        cx.method("combos_prepare (300 regs, 5 combos)", |cx, name| {
            let combo_count = 5;
            let reg_sizes: Vec<u32> = (0..300).map(|i| 1 + i % 3).collect();
            let reg_combo_ids: Vec<u32> = (0..300).map(|i| i % combo_count).collect();
            let taps: usize = reg_sizes.iter().map(|&s| s as usize).sum();
            let coeff_u = cx.exts(taps + MetalHal::<MH>::CHECK_SIZE);
            let mix = FpExt::random(&mut cx.rng);
            let combos = cx
                .hal
                .alloc_extelem_zeroed("combos", rows * (combo_count as usize + 1));
            cx.run(name, 0.0, 0.0, 0, || {
                cx.hal.combos_prepare(
                    &combos,
                    &coeff_u,
                    combo_count as usize,
                    rows,
                    &reg_sizes,
                    &reg_combo_ids,
                    &mix,
                )
            });
        });
        cx.method("combos_divide (6 combos, 11 roots)", |cx, name| {
            let v = cx.exts(7 * rows);
            let combos = cx.hal.copy_from_extelem("combos", &v);
            let roots = [2usize, 2, 2, 2, 2, 1];
            let chunks: Vec<(usize, Vec<FpExt>)> = roots
                .iter()
                .enumerate()
                .map(|(c, &r)| (c, cx.exts(r)))
                .collect();
            // Random data: nonzero remainders are logged through `tracing` (no subscriber here).
            cx.run(name, 2.0 * 16.0 * (6 * rows) as f64, 0.0, 0, || {
                cx.hal.combos_divide(&combos, chunks.clone(), rows)
            });
        });
        cx.method("eltwise_sum_extelem (6 combos)", |cx, name| {
            let v = cx.exts(6 * rows);
            let input = cx.hal.copy_from_extelem("input", &v);
            let out = cx.hal.alloc_elem("out", 4 * rows);
            cx.run(name, 16.0 * (7 * rows) as f64, 0.0, 0, || {
                cx.hal.eltwise_sum_extelem(&out, &input)
            });
        });
        cx.method("fri_fold (first round)", |cx, name| {
            let count = rows * INV_RATE / 16;
            let v = cx.elems(4 * 16 * count);
            let input = cx.hal.copy_from_elem("input", &v);
            let out = cx.hal.alloc_elem("out", 4 * count);
            let mix = FpExt::random(&mut cx.rng);
            cx.run(name, w * (68 * count) as f64, 0.0, 0, || {
                cx.hal.fri_fold(&out, &input, &mix)
            });
        });
        cx.method("scatter (4 entries/cycle)", |cx, name| {
            let index: Vec<u32> = (0..=rows as u32).map(|c| 4 * c).collect();
            let offsets: Vec<u32> = (0..4 * rows as u32)
                .map(|i| (i % 4) * rows as u32 + i / 4)
                .collect();
            let values = cx.elems(4 * rows);
            let into = cx.hal.alloc_elem("into", 4 * rows);
            cx.run(name, w * (8 * rows) as f64, 0.0, 0, || {
                cx.hal.scatter(&into, &index, &offsets, &values)
            });
        });
        cx.method("gather_sample", |cx, name| {
            let v = cx.elems(n);
            let src = cx.hal.copy_from_elem("src", &v);
            let dst = cx.hal.alloc_elem("dst", cols);
            cx.run(name, 0.0, 0.0, 0, || {
                cx.hal.gather_sample(&dst, &src, 7, cols, rows)
            });
        });
    }

    fn header<MH: MetalHash>(hal: &MetalHal<MH>, hash: &str, iters: usize) {
        println!(
            "# Metal HAL bench: device {}, hash {hash}, iters {iters}, working-set limit {} MiB \
             (recommended {} MiB)",
            hal.device.name(),
            hal.working_set_limit() >> 20,
            hal.device.recommended_max_working_set_size() >> 20,
        );
        println!(
            "{:>3} {:>4}  {:<34} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "po2", "cols", "method", "min ms", "med ms", "GB/s", "Mperm/s", "peak MiB"
        );
    }

    fn run_all<MH: MetalHash>(hal: MetalHal<MH>, hash: &str) {
        let iters = env_usize("RISC0_METAL_BENCH_ITERS", 5).max(1);
        let po2s = env_list("RISC0_METAL_BENCH_PO2", "16,18,20");
        let cols = env_list("RISC0_METAL_BENCH_COLS", "211");
        header(&hal, hash, iters);
        for &po2 in &po2s {
            for &c in &cols {
                run_shape(&hal, po2, c, iters);
            }
        }
    }

    pub fn main() {
        match std::env::var("RISC0_METAL_BENCH_HASH").as_deref() {
            Ok("sha256") => run_all(MetalHalSha256::new(), "sha256"),
            _ => run_all(MetalHalPoseidon2::new(), "poseidon2"),
        }
    }
}
