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

//! rv32im eval_check on the Metal HAL, and the rest of `CircuitHal`.
//!
//! R7 placeholder: eval_check delegates to the CPU constraint polynomial
//! (`risc0_circuit_rv32im_cpu_poly_fp`, the loop of `hal/cpu.rs`) over host views of the Metal
//! HAL's shared buffers. R10 replaces it with the baseline Metal kernel
//! (`ctx.require(Phase::EvalCheck)`); `eval_check_dep` stays the default no-op (one queue).

use rayon::prelude::*;
use risc0_circuit_rv32im_sys::risc0_circuit_rv32im_cpu_poly_fp;
use risc0_core::{
    field::{map_pow, Elem, ExtElem as _, RootsOfUnity as _},
    scope,
};
use risc0_zkp::{
    core::log2_ceil,
    hal::{
        metal::{BufferImpl, MetalHal, MetalHash},
        AccumPreflight, Buffer as _, CircuitHal,
    },
    INV_RATE,
};

use super::{count, host_slice, MetalCircuitHal, Phase};
use crate::{
    prove::{GLOBAL_MIX, GLOBAL_OUT},
    zirgen::{
        circuit::{ExtVal, Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_DATA},
        info::POLY_MIX_POWERS,
    },
};

impl<MH: MetalHash> CircuitHal<MetalHal<MH>> for MetalCircuitHal<MH> {
    fn eval_check(
        &self,
        check: &BufferImpl<Val>,
        groups: &[&BufferImpl<Val>],
        globals: &[&BufferImpl<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) {
        scope!("metal_eval_check(cpu)");
        count(Phase::EvalCheck);
        self.sync().unwrap();

        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        tracing::debug!("eval_check: domain {domain} (CPU kernels over Metal buffers)");
        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);

        // SAFETY: synced, and nothing is encoded until this call returns.
        let (data, accum, mix, out) = unsafe {
            (
                host_slice(groups[REGISTER_GROUP_DATA]),
                host_slice(groups[REGISTER_GROUP_ACCUM]),
                host_slice(globals[GLOBAL_MIX]),
                host_slice(globals[GLOBAL_OUT]),
            )
        };
        assert_eq!(check.size(), ExtVal::EXT_SIZE * domain);
        let check_ptr = check.as_mut_ptr() as usize;
        let poly_mix_pows = poly_mix_pows.as_slice();
        let args: &[&[Val]] = &[accum, data, out, mix];

        (0..domain).into_par_iter().for_each(|cycle| {
            let args: Vec<*const Val> = args.iter().map(|x| (*x).as_ptr()).collect();
            let mut tot = ExtVal::ZERO;
            unsafe {
                risc0_circuit_rv32im_cpu_poly_fp(
                    cycle,
                    domain,
                    poly_mix_pows.as_ptr(),
                    args.as_ptr(),
                    &mut tot,
                )
            };
            let x = Val::ROU_FWD[po2 + EXP_PO2].pow(cycle);
            // The 3 is the coset shift of the evaluation domain (as hal/cpu.rs).
            let y = (Val::new(3) * x).pow(1 << po2);
            let ret = tot * (y - Val::new(1)).inv();
            let check = check_ptr as *mut Val;
            for i in 0..ExtVal::EXT_SIZE {
                // SAFETY: in bounds (asserted above); each cycle writes disjoint cells.
                unsafe { *check.add(i * domain + cycle) = ret.elems()[i] };
            }
        });
    }

    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        _ctrl: &BufferImpl<Val>,
        _io: &BufferImpl<Val>,
        _data: &BufferImpl<Val>,
        _mix: &BufferImpl<Val>,
        _accum: &BufferImpl<Val>,
        _steps: usize,
    ) {
        // As hal/cuda.rs: rv32im accumulates in CircuitAccumulator::step_accum (PLAN 8.4).
    }
}
