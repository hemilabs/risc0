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

//! Recursion eval_check on the Metal HAL (and the zkp `CircuitHal` impl), item C8.
//!
//! Host side of `ffi_supra.cu:73-117` (`risc0_circuit_recursion_cuda_eval_check`): the 158
//! poly_mix powers go into a HAL-owned shared buffer (CUDA: `cudaMemcpyToSymbolAsync` into
//! `__constant__ FpExt poly_mix[158]`), then the kernels C3 translated run over the evaluation
//! domain on the open batch:
//! - `lz` (default since C12): `ec_lz_part_0..5`, the split below with every part's Fp sums of
//!   products accumulated lazily (`cuda2metal_eval.py --split-lz`, ec_lazyfp.py's `lz2` form);
//!   the same bindings and dispatch order as the split;
//! - `split` (`RISC0_METAL_REC_KERNELS=split`, the C11 default): `ec_part_0..5` back to back
//!   over the same bindings on the serial encoder (`eval_check_split.cu` `launch_parts`; part
//!   P > 0 adds to the sum part P-1 left in `check`, part 5 applies the quotient);
//! - `monolith` (`RISC0_METAL_REC_KERNELS=monolith`, the pre-C11 default): one `eval_check`
//!   dispatch (`eval_check.cu` + `ffi_supra.cu:43-68`).
//!
//! All three are bit-identical to the CPU `poly_fp` (tests below). The batch is flushed, not waited
//! for, so the prover's `post_eval_check` host work overlaps the GPU (CUDA: async on the stream).
//! eval_check has no asserts and every trace read is `& mask`-clamped, so there is no fault word;
//! the host checks the buffer sizes the kernel relies on.
//!
//! `RISC0_METAL_EVAL_CHECK_SHADOW=1` re-runs every eval_check on the CPU after the GPU and panics
//! on the first differing word (debug aid for real lift/join traces; costs a sync + ~1 s at po2
//! 18).

use std::sync::OnceLock;

use metal::ComputePipelineState;
use risc0_circuit_recursion_sys::{risc0_circuit_recursion_cpu_eval_check, RawAccumBuffers};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    adapter::{CircuitInfo as _, TapsProvider as _},
    core::log2_ceil,
    field::{
        baby_bear::{BabyBearElem, BabyBearExtElem},
        map_pow, Elem as _, ExtElem as _, RootsOfUnity as _,
    },
    hal::{
        metal::{as_bytes, BufferImpl, KernelArg, MetalHal, MetalHash},
        AccumPreflight, Buffer as _, CircuitHal,
    },
    INV_RATE,
};

use super::{grow, host_slice, EvalKernels, MetalCircuitHal, Phase, Pipelines};
use crate::{
    CircuitImpl, GLOBAL_MIX, GLOBAL_OUT, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CTRL,
    REGISTER_GROUP_DATA,
};

/// Debug knob: compare every Metal eval_check against the CPU one.
pub(crate) const SHADOW_ENV: &str = "RISC0_METAL_EVAL_CHECK_SHADOW";

/// `RecEvalParams` of the generated `eval_check.metal` / `eval_check_split.metal` preludes
/// (`static_assert(sizeof(RecEvalParams) == 16)` there), bound at buffer 7 with `set_bytes`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RecEvalParams {
    /// Root of unity of the evaluation domain, raw Montgomery word (`Fp::fromRaw`).
    pub rou: u32,
    /// Trace po2.
    pub po2: u32,
    /// Evaluation domain size (`steps * INV_RATE`, a power of two).
    pub domain: u32,
    pub _pad: u32,
}
const _: () = assert!(std::mem::size_of::<RecEvalParams>() == 16);
// SAFETY: `repr(C)`, four `u32` fields, no padding (size asserted above).
unsafe impl bytemuck::NoUninit for RecEvalParams {}

/// Kernel ABI (C3 note, "Kernel ABI"): buffer indices of both variants.
mod bind {
    pub const POLY_MIX: usize = 0;
    pub const CHECK: usize = 1;
    pub const CTRL: usize = 2;
    pub const DATA: usize = 3;
    pub const ACCUM: usize = 4;
    pub const MIX: usize = 5;
    pub const OUT: usize = 6;
    pub const PARAMS: usize = 7;
    pub const COUNT: usize = 8;
}

/// Whether [SHADOW_ENV] is set (read once).
fn shadow_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var(SHADOW_ENV).is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// The sizes the kernels index without bounds checks (beyond `& mask`), checked on the host.
fn check_shapes(
    check: &BufferImpl<BabyBearElem>,
    ctrl: &BufferImpl<BabyBearElem>,
    data: &BufferImpl<BabyBearElem>,
    accum: &BufferImpl<BabyBearElem>,
    mix: &BufferImpl<BabyBearElem>,
    out: &BufferImpl<BabyBearElem>,
    domain: usize,
) {
    assert!(
        domain.is_power_of_two(),
        "eval_check: domain {domain} is not a power of two"
    );
    assert!(
        domain <= u32::MAX as usize,
        "eval_check: domain {domain} exceeds u32"
    );
    let taps = CircuitImpl.get_taps();
    assert_eq!(
        check.size(),
        BabyBearExtElem::EXT_SIZE * domain,
        "eval_check: check size"
    );
    for (name, buf, group) in [
        ("ctrl", ctrl, REGISTER_GROUP_CTRL),
        ("data", data, REGISTER_GROUP_DATA),
        ("accum", accum, REGISTER_GROUP_ACCUM),
    ] {
        let need = taps.group_size(group) * domain;
        assert!(
            buf.size() >= need,
            "eval_check: {name} has {} < {need} elements",
            buf.size()
        );
    }
    assert!(
        mix.size() >= CircuitImpl::MIX_SIZE,
        "eval_check: mix size {}",
        mix.size()
    );
    assert!(
        out.size() >= CircuitImpl::OUTPUT_SIZE,
        "eval_check: out size {}",
        out.size()
    );
}

/// The CPU eval_check (`hal/cpu.rs`, `kernels/cxx/poly_fp.cpp`) into a new vector: the reference
/// for the shadow check and the tests. The inputs must be synced host views.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_eval_check(
    ctrl: &[BabyBearElem],
    data: &[BabyBearElem],
    accum: &[BabyBearElem],
    mix: &[BabyBearElem],
    out: &[BabyBearElem],
    poly_mix_pows: &[BabyBearExtElem],
    rou: BabyBearElem,
    po2: usize,
    domain: usize,
) -> Vec<BabyBearElem> {
    let check = vec![BabyBearElem::ZERO; BabyBearExtElem::EXT_SIZE * domain];
    let buffers = RawAccumBuffers {
        ctrl: ctrl.as_ptr(),
        global: out.as_ptr(),
        data: data.as_ptr(),
        mix: mix.as_ptr(),
        accum: accum.as_ptr(),
    };
    // The C++ writes through `check` (a const pointer in the FFI signature, as hal/cpu.rs).
    ffi_wrap(|| unsafe {
        risc0_circuit_recursion_cpu_eval_check(
            &buffers,
            poly_mix_pows.as_ptr(),
            check.as_ptr(),
            rou,
            po2 as u32,
            domain as u32,
        )
    })
    .unwrap();
    check
}

impl<MH: MetalHash> MetalCircuitHal<MH> {
    /// Encode eval_check with an explicit variant and its PSOs into the open batch (no flush).
    /// `poly_mix_pows` must hold the 158 powers. Tests call it with both variants.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_eval_check(
        &self,
        variant: EvalKernels,
        kernels: &Pipelines,
        check: &BufferImpl<BabyBearElem>,
        groups: &[&BufferImpl<BabyBearElem>],
        globals: &[&BufferImpl<BabyBearElem>],
        poly_mix_pows: &[BabyBearExtElem],
        po2: usize,
        steps: usize,
    ) {
        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        let (ctrl, data, accum) = (
            groups[REGISTER_GROUP_CTRL],
            groups[REGISTER_GROUP_DATA],
            groups[REGISTER_GROUP_ACCUM],
        );
        let (mix, out) = (globals[GLOBAL_MIX], globals[GLOBAL_OUT]);
        check_shapes(check, ctrl, data, accum, mix, out, domain);
        assert_eq!(poly_mix_pows.len(), crate::info::NUM_POLY_MIX_POWERS);

        // poly_mix: HAL-owned scratch (PLAN 2.6). `view_mut` syncs first if any batch is
        // unsynced, so a previous eval_check still reading it is never overwritten.
        let poly_mix = grow(
            &self.hal,
            &mut self.scratch().poly_mix,
            "rec_poly_mix",
            poly_mix_pows.len(),
        );
        poly_mix.view_mut(|v| v.copy_from_slice(poly_mix_pows));
        self.record(Phase::EvalCheck);

        let params = [RecEvalParams {
            rou: BabyBearElem::ROU_FWD[po2 + EXP_PO2].as_u32_montgomery(),
            po2: po2 as u32,
            domain: domain as u32,
            _pad: 0,
        }];
        let mut args: [KernelArg; bind::COUNT] = std::array::from_fn(|_| KernelArg::Null);
        args[bind::POLY_MIX] = poly_mix.as_arg();
        args[bind::CHECK] = check.as_arg();
        args[bind::CTRL] = ctrl.as_arg();
        args[bind::DATA] = data.as_arg();
        args[bind::ACCUM] = accum.as_arg();
        args[bind::MIX] = mix.as_arg();
        args[bind::OUT] = out.as_arg();
        args[bind::PARAMS] = KernelArg::Bytes(as_bytes(&params));

        let psos: Vec<&ComputePipelineState> = variant
            .names()
            .iter()
            .map(|n| kernels.pipeline(n))
            .collect();
        for pso in psos {
            // 1-D over the domain, threadgroups of min(256, PSO max) (`getSimpleConfig`) unless
            // RISC0_METAL_REC_EVAL_TG says otherwise (C11 sweep; the domain is a power of two,
            // so every width divides it).
            let grid = super::grid_1d(super::TgKind::Eval, pso, domain as u64);
            self.hal.dispatch(pso, &args, grid, &[]);
        }
    }

    /// Sync, then compare `check` with the CPU eval_check over the same (synced) inputs; panics on
    /// the first differing word.
    fn shadow_check(
        &self,
        check: &BufferImpl<BabyBearElem>,
        groups: &[&BufferImpl<BabyBearElem>],
        globals: &[&BufferImpl<BabyBearElem>],
        poly_mix_pows: &[BabyBearExtElem],
        po2: usize,
        steps: usize,
    ) {
        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        self.sync().unwrap();
        // SAFETY: synced, and nothing is encoded until this returns.
        let (ctrl, data, accum, mix, out, gpu) = unsafe {
            (
                host_slice(groups[REGISTER_GROUP_CTRL]),
                host_slice(groups[REGISTER_GROUP_DATA]),
                host_slice(groups[REGISTER_GROUP_ACCUM]),
                host_slice(globals[GLOBAL_MIX]),
                host_slice(globals[GLOBAL_OUT]),
                host_slice(check),
            )
        };
        let rou = BabyBearElem::ROU_FWD[po2 + EXP_PO2];
        let cpu = cpu_eval_check(ctrl, data, accum, mix, out, poly_mix_pows, rou, po2, domain);
        if let Some(i) = (0..cpu.len()).find(|&i| cpu[i] != gpu[i]) {
            panic!(
                "{SHADOW_ENV}: Metal eval_check ({:?}) differs from the CPU at po2 {po2}, word {i} \
                 (column {}, cycle {}): metal {:?} cpu {:?}",
                EvalKernels::selected(),
                i / domain,
                i % domain,
                gpu[i],
                cpu[i]
            );
        }
        // stderr, not tracing: the knob is a debug aid whose confirmation must be visible.
        eprintln!(
            "{SHADOW_ENV}: eval_check ({:?}) po2 {po2} matches the CPU ({} words)",
            EvalKernels::selected(),
            cpu.len()
        );
    }
}

/// Test hook (C13): run this thread's eval_check calls with another variant than the process
/// selected, for in-process interleaved A/B of two kernel forms on whole proofs.
#[cfg(all(test, feature = "test"))]
pub(crate) mod test_override {
    use std::{cell::Cell, sync::OnceLock};

    use super::*;

    thread_local! {
        static VARIANT: Cell<Option<EvalKernels>> = const { Cell::new(None) };
    }

    /// Run `f` with this thread's eval_check variant forced to `v`.
    pub(crate) fn with_variant<R>(v: EvalKernels, f: impl FnOnce() -> R) -> R {
        let old = VARIANT.with(|c| c.replace(Some(v)));
        struct Restore(Option<EvalKernels>);
        impl Drop for Restore {
            fn drop(&mut self) {
                VARIANT.with(|c| c.set(self.0));
            }
        }
        let _restore = Restore(old);
        f()
    }

    pub(super) fn eval_kernels<MH: MetalHash>(
        c: &MetalCircuitHal<MH>,
        selected: EvalKernels,
        kernels: &'static Pipelines,
    ) -> (EvalKernels, &'static Pipelines) {
        let Some(v) = VARIANT.with(|c| c.get()).filter(|&v| v != selected) else {
            return (selected, kernels);
        };
        static OTHER: [OnceLock<Pipelines>; EvalKernels::ALL.len()] =
            [const { OnceLock::new() }; EvalKernels::ALL.len()];
        let k = OTHER[v as usize].get_or_init(|| {
            Pipelines::new(
                &c.hal().device,
                risc0_circuit_recursion_sys::METAL_LIB,
                v.names(),
                4,
            )
            .unwrap()
        });
        (v, k)
    }
}

impl<MH: MetalHash> CircuitHal<MetalHal<MH>> for MetalCircuitHal<MH> {
    fn eval_check(
        &self,
        check: &BufferImpl<BabyBearElem>,
        groups: &[&BufferImpl<BabyBearElem>],
        globals: &[&BufferImpl<BabyBearElem>],
        poly_mix: BabyBearExtElem,
        po2: usize,
        steps: usize,
    ) {
        scope!("metal_eval_check");
        self.enter(Phase::EvalCheck);
        let poly_mix_pows = map_pow(poly_mix, crate::info::POLY_MIX_POWERS);
        // Built at warmup or on first use (cold: tens of seconds; never per proof).
        let (variant, kernels) = self
            .eval_kernels()
            .unwrap_or_else(|e| panic!("recursion Metal eval_check: {e:#}"));
        #[cfg(all(test, feature = "test"))]
        let (variant, kernels) = test_override::eval_kernels(self, variant, kernels);
        self.encode_eval_check(
            variant,
            kernels,
            check,
            groups,
            globals,
            &poly_mix_pows,
            po2,
            steps,
        );
        if shadow_enabled() {
            self.shadow_check(check, groups, globals, &poly_mix_pows, po2, steps);
        }
        // End of the phase (PLAN 2.6): commit without waiting; readers of `check` sync.
        self.hal.flush();
        if super::profile::enabled() {
            self.prof_mark("eval_check");
            super::profile::report();
        }
    }

    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        _ctrl: &BufferImpl<BabyBearElem>,
        _io: &BufferImpl<BabyBearElem>,
        _data: &BufferImpl<BabyBearElem>,
        _mix: &BufferImpl<BabyBearElem>,
        _accum: &BufferImpl<BabyBearElem>,
        _steps: usize,
    ) {
        // As in cuda.rs/hip.rs: recursion accumulates through `CircuitAccumulator`.
        unimplemented!()
    }

    /// Round 1 runs eval_check on the one command queue (PLAN 2.0, errata 8.4): nothing to wait
    /// for.
    fn eval_check_dep(&self) {}
}

#[cfg(all(test, feature = "test"))]
mod c8_tests {
    use std::{sync::Arc, time::Instant};

    use risc0_core::field::baby_bear::BabyBear;
    use risc0_zkp::{
        core::hash::sha::Sha256HashSuite,
        hal::{
            cpu::CpuHal,
            metal::{MetalHalPoseidon2, MetalHalSha256, MetalHashPoseidon2},
            Hal as _,
        },
    };
    use test_log::test;

    use super::*;
    use crate::{
        prove::hal::cpu::CpuCircuitHal,
        testutil::{eval_check, EvalCheckParams},
    };

    /// The PSOs of `variant`, whichever variant `RISC0_METAL_REC_KERNELS` selected for the
    /// process (the others are built here once each; cold ~10-60 s the first time on a machine).
    fn variant_kernels(
        c: &MetalCircuitHal<MetalHashPoseidon2>,
        variant: EvalKernels,
    ) -> &'static Pipelines {
        let (selected, kernels) = c.eval_kernels().unwrap();
        if selected == variant {
            return kernels;
        }
        static OTHER: [OnceLock<Pipelines>; EvalKernels::ALL.len()] =
            [const { OnceLock::new() }; EvalKernels::ALL.len()];
        OTHER[variant as usize].get_or_init(|| {
            let t0 = Instant::now();
            let k = Pipelines::new(
                &c.hal().device,
                risc0_circuit_recursion_sys::METAL_LIB,
                variant.names(),
                4,
            )
            .unwrap();
            eprintln!(
                "C8: {variant:?} PSOs built in {:.2}s",
                t0.elapsed().as_secs_f64()
            );
            k
        })
    }

    /// `testutil::eval_check` at PO2 4: CPU circuit HAL vs the Metal one (selected variant).
    #[test]
    fn eval_check_po2_4_matches_cpu() {
        let cpu_hal: CpuHal<BabyBear> = CpuHal::new(Sha256HashSuite::new_suite());
        let gpu_hal = Arc::new(MetalHalSha256::new());
        let gpu_eval = MetalCircuitHal::new(gpu_hal.clone()).unwrap();
        eval_check(&cpu_hal, CpuCircuitHal, gpu_hal.as_ref(), gpu_eval, 4);
    }

    /// Every variant (monolith, split, lazy Fp split with uint4 and with four-load mix reads) on
    /// the same random inputs, each bit-identical to the CPU eval_check.
    /// Prints the warm wall time of encode + sync per variant (indicative GPU time) and the peak
    /// device allocation of the step.
    fn variants_match_cpu(po2: usize) {
        let _guard = crate::prove::hal::metal::global_state_test_lock();
        let hal = Arc::new(MetalHalPoseidon2::new());
        let c = MetalCircuitHal::new(hal.clone()).unwrap();
        let variants: Vec<(EvalKernels, &Pipelines)> = EvalKernels::ALL
            .iter()
            .map(|&v| (v, variant_kernels(&c, v)))
            .collect();

        let p = EvalCheckParams::new(po2);
        let poly_mix_pows = map_pow(p.poly_mix, crate::info::POLY_MIX_POWERS);
        let rou = BabyBearElem::ROU_FWD[po2 + log2_ceil(INV_RATE)];
        let t0 = Instant::now();
        let want = cpu_eval_check(
            &p.code,
            &p.data,
            &p.accum,
            &p.mix,
            &p.out,
            &poly_mix_pows,
            rou,
            po2,
            p.domain,
        );
        let cpu_s = t0.elapsed().as_secs_f64();

        hal.sync();
        hal.reset_peak_allocated_size();
        let before = hal.allocated_size();
        let code = hal.copy_from_elem("code", &p.code);
        let data = hal.copy_from_elem("data", &p.data);
        let accum = hal.copy_from_elem("accum", &p.accum);
        let mix = hal.copy_from_elem("mix", &p.mix);
        let out = hal.copy_from_elem("out", &p.out);
        let groups = [&accum, &code, &data];
        let globals = [&mix, &out];

        let mut results = Vec::new();
        for (variant, kernels) in variants {
            let check = hal.alloc_elem("check", BabyBearExtElem::EXT_SIZE * p.domain);
            // Dirty the output so a missing write cannot pass.
            check.view_mut(|v| v.fill(BabyBearElem::new(12345)));
            let mut times = Vec::new();
            for _ in 0..3 {
                let t0 = Instant::now();
                c.encode_eval_check(
                    variant,
                    kernels,
                    &check,
                    &groups,
                    &globals,
                    &poly_mix_pows,
                    po2,
                    p.steps,
                );
                hal.sync();
                times.push(t0.elapsed().as_secs_f64() * 1e3);
            }
            let got = check.to_vec();
            if let Some(i) = (0..want.len()).find(|&i| want[i] != got[i]) {
                panic!(
                    "{variant:?} po2 {po2}: word {i} (column {}, cycle {}): metal {:?} cpu {:?}",
                    i / p.domain,
                    i % p.domain,
                    got[i],
                    want[i]
                );
            }
            results.push((variant, times));
        }
        let peak = hal.peak_allocated_size();
        for (variant, times) in &results {
            let best = times.iter().cloned().fold(f64::MAX, f64::min);
            eprintln!(
                "C8 po2 {po2} (domain {}): {variant:?} == CPU; encode+sync ms {times:.2?} (best \
                 {best:.2})",
                p.domain
            );
        }
        eprintln!(
            "C8 po2 {po2}: CPU eval_check {:.1} ms; device alloc before inputs {before} B, peak \
             {peak} B (+{} B), limit {} B",
            cpu_s * 1e3,
            peak.saturating_sub(before),
            hal.working_set_limit()
        );
    }

    #[test]
    fn variants_match_cpu_po2_4() {
        variants_match_cpu(4);
    }

    #[test]
    fn variants_match_cpu_po2_10() {
        variants_match_cpu(10);
    }

    #[test]
    fn variants_match_cpu_po2_16() {
        variants_match_cpu(16);
    }

    /// RECURSION_PO2: the shape of every lift/join/resolve/identity proof.
    #[test]
    fn variants_match_cpu_po2_18() {
        variants_match_cpu(18);
    }

    /// A too-small input group is a programming error caught on the host.
    #[test]
    #[should_panic(expected = "eval_check: data has")]
    fn short_group_panics() {
        let hal = Arc::new(MetalHalPoseidon2::new());
        let c = MetalCircuitHal::new(hal.clone()).unwrap();
        let p = EvalCheckParams::new(4);
        let pows = map_pow(p.poly_mix, crate::info::POLY_MIX_POWERS);
        let check = hal.alloc_elem("check", BabyBearExtElem::EXT_SIZE * p.domain);
        let code = hal.copy_from_elem("code", &p.code);
        let data = hal.copy_from_elem("data", &p.data[..p.data.len() - 1]);
        let accum = hal.copy_from_elem("accum", &p.accum);
        let mix = hal.copy_from_elem("mix", &p.mix);
        let out = hal.copy_from_elem("out", &p.out);
        let (variant, kernels) = c.eval_kernels().unwrap();
        c.encode_eval_check(
            variant,
            kernels,
            &check,
            &[&accum, &code, &data],
            &[&mix, &out],
            &pows,
            4,
            p.steps,
        );
    }
}
