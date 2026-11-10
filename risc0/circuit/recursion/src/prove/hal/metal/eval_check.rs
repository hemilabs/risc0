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

//! Recursion eval_check on the Metal HAL (and the zkp `CircuitHal` impl).
//!
//! C5 placeholder: runs the CPU eval_check (`risc0_circuit_recursion_cpu_eval_check`, as
//! `hal/cpu.rs` calls it) over host views of the Metal HAL's shared buffers, after a `sync()`. C8
//! replaces this with the Metal kernels selected by `RISC0_METAL_REC_KERNELS`
//! ([super::MetalCircuitHal::eval_kernels]).

use risc0_circuit_recursion_sys::{risc0_circuit_recursion_cpu_eval_check, RawAccumBuffers};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    core::log2_ceil,
    field::{
        baby_bear::{BabyBearElem, BabyBearExtElem},
        map_pow, RootsOfUnity as _,
    },
    hal::{
        metal::{BufferImpl, MetalHal, MetalHash},
        AccumPreflight, CircuitHal,
    },
    INV_RATE,
};

use super::{host_slice, MetalCircuitHal, Phase};
use crate::{
    GLOBAL_MIX, GLOBAL_OUT, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CTRL, REGISTER_GROUP_DATA,
};

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
        scope!("metal_eval_check(cpu)");
        self.enter(Phase::EvalCheck);
        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        let rou = BabyBearElem::ROU_FWD[po2 + EXP_PO2];
        let poly_mix_pows = map_pow(poly_mix, crate::info::POLY_MIX_POWERS);

        self.sync().unwrap();
        // SAFETY: synced, and nothing is encoded until this call returns.
        let (ctrl, data, accum, mix, global, check) = unsafe {
            (
                host_slice(groups[REGISTER_GROUP_CTRL]),
                host_slice(groups[REGISTER_GROUP_DATA]),
                host_slice(groups[REGISTER_GROUP_ACCUM]),
                host_slice(globals[GLOBAL_MIX]),
                host_slice(globals[GLOBAL_OUT]),
                host_slice(check),
            )
        };
        let buffers = RawAccumBuffers {
            ctrl: ctrl.as_ptr(),
            global: global.as_ptr(),
            data: data.as_ptr(),
            mix: mix.as_ptr(),
            accum: accum.as_ptr(),
        };
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
}
