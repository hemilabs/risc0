// Copyright 2025 RISC Zero, Inc.
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

use std::sync::Arc;

use anyhow::{bail, Result};
use risc0_circuit_recursion_sys::{
    risc0_circuit_recursion_cpu_accum, risc0_circuit_recursion_cpu_eval_check,
    risc0_circuit_recursion_cpu_witgen, RawAccumBuffers, RawExecBuffers, RawPreflightTrace,
    StepMode,
};
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    core::{
        hash::{
            poseidon2::Poseidon2HashSuite, poseidon_254::Poseidon254HashSuite, sha::Sha256HashSuite,
        },
        log2_ceil,
    },
    field::{
        baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
        map_pow, RootsOfUnity as _,
    },
    hal::{cpu::CpuBuffer, AccumPreflight, CircuitHal},
    INV_RATE,
};

use crate::{
    prove::{RecursionProver, RecursionProverImpl},
    GLOBAL_MIX, GLOBAL_OUT, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CTRL, REGISTER_GROUP_DATA,
};

use super::{CircuitAccumulator, CircuitWitnessGenerator};

type CpuHal = risc0_zkp::hal::cpu::CpuHal<BabyBear>;

/// Cycles per call of the 16-wide constraint polynomial.
const POLY_FP_LANES: usize = 16;

type PolyFpX16 = unsafe extern "C" fn(
    usize,
    usize,
    *const BabyBearExtElem,
    *const *const BabyBearElem,
    *mut BabyBearExtElem,
) -> *const std::os::raw::c_char;

/// The 16-wide constraint polynomial for this CPU (AVX-512, else AVX2), if the build has it.
/// `RISC0_CPU_EVAL_SCALAR` set forces the scalar one.
fn poly_fp_x16() -> Option<PolyFpX16> {
    #[cfg(target_arch = "x86_64")]
    {
        use risc0_circuit_recursion_sys::{
            risc0_circuit_recursion_cpu_poly_fp_x16_available,
            risc0_circuit_recursion_cpu_poly_fp_x16_avx2,
            risc0_circuit_recursion_cpu_poly_fp_x16_avx512,
        };
        if std::env::var_os("RISC0_CPU_EVAL_SCALAR").is_some()
            || !unsafe { risc0_circuit_recursion_cpu_poly_fp_x16_available() }
        {
            return None;
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            return Some(risc0_circuit_recursion_cpu_poly_fp_x16_avx512);
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return Some(risc0_circuit_recursion_cpu_poly_fp_x16_avx2);
        }
    }
    None
}

#[derive(Default)]
pub(crate) struct CpuCircuitHal;

impl CircuitWitnessGenerator<CpuHal> for CpuCircuitHal {
    fn generate_witness(
        &self,
        mode: StepMode,
        total_cycles: u32,
        preflight: &RawPreflightTrace,
        ctrl: &CpuBuffer<BabyBearElem>,
        data: &CpuBuffer<BabyBearElem>,
        global: &CpuBuffer<BabyBearElem>,
    ) -> Result<()> {
        let ctrl = ctrl.as_slice();
        let data = data.as_slice();
        let global = global.as_slice();
        let buffers = RawExecBuffers {
            ctrl: ctrl.as_ptr(),
            data: data.as_ptr(),
            global: global.as_ptr(),
        };
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_cpu_witgen(mode, &buffers, preflight, total_cycles)
        })
    }
}

impl CircuitAccumulator<CpuHal> for CpuCircuitHal {
    fn accumulate(
        &self,
        work_cycles: u32,
        total_cycles: u32,
        ctrl: &CpuBuffer<BabyBearElem>,
        global: &CpuBuffer<BabyBearElem>,
        data: &CpuBuffer<BabyBearElem>,
        mix: &CpuBuffer<BabyBearElem>,
        accum: &CpuBuffer<BabyBearElem>,
    ) -> Result<()> {
        let ctrl = ctrl.as_slice();
        let global = global.as_slice();
        let data = data.as_slice();
        let mix = mix.as_slice();
        let accum = accum.as_slice();
        let buffers = RawAccumBuffers {
            ctrl: ctrl.as_ptr(),
            global: global.as_ptr(),
            data: data.as_ptr(),
            mix: mix.as_ptr(),
            accum: accum.as_ptr(),
        };
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_cpu_accum(&buffers, work_cycles, total_cycles)
        })
    }
}

impl CircuitHal<CpuHal> for CpuCircuitHal {
    fn eval_check(
        &self,
        check: &CpuBuffer<BabyBearElem>,
        groups: &[&CpuBuffer<BabyBearElem>],
        globals: &[&CpuBuffer<BabyBearElem>],
        poly_mix: BabyBearExtElem,
        po2: usize,
        steps: usize,
    ) {
        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;

        let ctrl = groups[REGISTER_GROUP_CTRL].as_slice();
        let data = groups[REGISTER_GROUP_DATA].as_slice();
        let accum = groups[REGISTER_GROUP_ACCUM].as_slice();
        let mix = globals[GLOBAL_MIX].as_slice();
        let global = globals[GLOBAL_OUT].as_slice();
        let check = check.as_slice();

        let poly_mix_pows = map_pow(poly_mix, crate::info::POLY_MIX_POWERS);
        let poly_mix_pows = poly_mix_pows.as_slice();

        let rou = BabyBearElem::ROU_FWD[po2 + EXP_PO2];

        let buffers = RawAccumBuffers {
            ctrl: ctrl.as_ptr(),
            global: global.as_ptr(),
            data: data.as_ptr(),
            mix: mix.as_ptr(),
            accum: accum.as_ptr(),
        };

        if let Some(poly_fp_x16) = poly_fp_x16().filter(|_| domain % POLY_FP_LANES == 0) {
            use rayon::prelude::*;
            use risc0_zkp::field::{Elem as _, ExtElem as _};

            let args: [*const BabyBearElem; 5] = [
                ctrl.as_ptr(),
                global.as_ptr(),
                data.as_ptr(),
                mix.as_ptr(),
                accum.as_ptr(),
            ];
            // Shared across threads as addresses: raw pointers are not Send.
            let args_addr = args.as_ptr() as usize;
            let check_addr = check.as_ptr() as usize;
            let check_len = check.len();
            // The polynomial is divided by (3x)^(2^po2) - 1 for x = rou^cycle; rou^(2^po2) has
            // order 4, so that depends only on cycle mod 4.
            let inv_zerofier: [BabyBearElem; 4] = std::array::from_fn(|k| {
                let x = rou.pow(k);
                ((BabyBearElem::new(3) * x).pow(1 << po2) - BabyBearElem::new(1)).inv()
            });
            (0..domain / POLY_FP_LANES)
                .into_par_iter()
                .for_each(|block| {
                    let cycle = block * POLY_FP_LANES;
                    let mut tots = [BabyBearExtElem::ZERO; POLY_FP_LANES];
                    ffi_wrap(|| unsafe {
                        poly_fp_x16(
                            cycle,
                            domain,
                            poly_mix_pows.as_ptr(),
                            args_addr as *const *const BabyBearElem,
                            tots.as_mut_ptr(),
                        )
                    })
                    .expect("eval_check");
                    // SAFETY: each block writes only its own cycles.
                    let check = unsafe {
                        std::slice::from_raw_parts_mut(check_addr as *mut BabyBearElem, check_len)
                    };
                    for (lane, tot) in tots.into_iter().enumerate() {
                        let ret = tot * inv_zerofier[(cycle + lane) % 4];
                        for i in 0..BabyBearExtElem::EXT_SIZE {
                            check[i * domain + cycle + lane] = ret.elems()[i];
                        }
                    }
                });
            return;
        }

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

    #[allow(unused)]
    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        ctrl: &CpuBuffer<BabyBearElem>,
        io: &CpuBuffer<BabyBearElem>,
        data: &CpuBuffer<BabyBearElem>,
        mix: &CpuBuffer<BabyBearElem>,
        accum: &CpuBuffer<BabyBearElem>,
        steps: usize,
    ) {
        unimplemented!()
    }
}

#[allow(dead_code)]
pub(crate) fn recursion_prover(hashfn: &str) -> Result<Box<dyn RecursionProver>> {
    let suite = match hashfn {
        "poseidon2" => Poseidon2HashSuite::new_suite(),
        "poseidon_254" => Poseidon254HashSuite::new_suite(),
        "sha-256" => Sha256HashSuite::new_suite(),
        _ => bail!("Unsupported hashfn: {hashfn}"),
    };

    let hal = Arc::new(CpuHal::new(suite));
    let circuit_hal = Arc::new(CpuCircuitHal);
    Ok(Box::new(RecursionProverImpl::new(hal, circuit_hal)))
}
