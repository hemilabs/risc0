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

use std::rc::Rc;

use anyhow::{bail, Result};
use risc0_circuit_recursion_sys::{
    risc0_circuit_recursion_cpu_accum, risc0_circuit_recursion_cpu_witgen,
    RawAccumBuffers, RawExecBuffers, RawPreflightTrace,
    StepMode,
};
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    core::{
        log2_ceil,
    },
    field::{
        baby_bear::{BabyBearElem, BabyBearExtElem},
        map_pow, RootsOfUnity as _,
    },
    hal::{
        intel::{
            BufferImpl as IntelBuffer, IntelHal, IntelHalPoseidon2,
            IntelHalPoseidon254, IntelHalSha256, IntelHash,
            IntelHashPoseidon2, IntelHashPoseidon254, IntelHashSha256,
        },
        AccumPreflight, Buffer, CircuitHal,
    },
    INV_RATE,
};

use crate::{
    prove::{RecursionProver, RecursionProverImpl},
    GLOBAL_MIX, GLOBAL_OUT, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CTRL, REGISTER_GROUP_DATA,
};

use super::{CircuitAccumulator, CircuitWitnessGenerator};

struct IntelCircuitHal<IH: IntelHash> {
    _hal: Rc<IntelHal<IH>>, // retain a reference to ensure the context remains valid
}

impl<IH: IntelHash> IntelCircuitHal<IH> {
    pub fn new(_hal: Rc<IntelHal<IH>>) -> Self {
        Self { _hal }
    }
}

impl<IH: IntelHash> CircuitWitnessGenerator<IntelHal<IH>> for IntelCircuitHal<IH> {
    fn generate_witness(
        &self,
        mode: StepMode,
        total_cycles: u32,
        preflight: &RawPreflightTrace,
        ctrl: &IntelBuffer<BabyBearElem>,
        data: &IntelBuffer<BabyBearElem>,
        global: &IntelBuffer<BabyBearElem>,
    ) -> Result<()> {
        // Download GPU buffers to host for CPU FFI
        let ctrl_host = ctrl.to_vec();
        let data_host = data.to_vec();
        let global_host = global.to_vec();

        let buffers = RawExecBuffers {
            ctrl: ctrl_host.as_ptr(),
            data: data_host.as_ptr(),
            global: global_host.as_ptr(),
        };
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_cpu_witgen(mode, &buffers, preflight, total_cycles)
        })?;

        // Upload results back to GPU (H2D only, no redundant D2H)
        ctrl.copy_from_host(&ctrl_host);
        data.copy_from_host(&data_host);
        global.copy_from_host(&global_host);

        Ok(())
    }
}

impl<IH: IntelHash> CircuitAccumulator<IntelHal<IH>> for IntelCircuitHal<IH> {
    fn accumulate(
        &self,
        work_cycles: u32,
        total_cycles: u32,
        ctrl: &IntelBuffer<BabyBearElem>,
        global: &IntelBuffer<BabyBearElem>,
        data: &IntelBuffer<BabyBearElem>,
        mix: &IntelBuffer<BabyBearElem>,
        accum: &IntelBuffer<BabyBearElem>,
    ) -> Result<()> {
        // Download GPU buffers to host for CPU FFI
        let ctrl_host = ctrl.to_vec();
        let global_host = global.to_vec();
        let data_host = data.to_vec();
        let mix_host = mix.to_vec();
        let accum_host = accum.to_vec();

        let buffers = RawAccumBuffers {
            ctrl: ctrl_host.as_ptr(),
            global: global_host.as_ptr(),
            data: data_host.as_ptr(),
            mix: mix_host.as_ptr(),
            accum: accum_host.as_ptr(),
        };
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_cpu_accum(&buffers, work_cycles, total_cycles)
        })?;

        // Upload modified buffers back to GPU (H2D only, no redundant D2H)
        accum.copy_from_host(&accum_host);
        global.copy_from_host(&global_host);

        Ok(())
    }
}

impl<IH: IntelHash> CircuitHal<IntelHal<IH>> for IntelCircuitHal<IH> {
    fn eval_check(
        &self,
        check: &IntelBuffer<BabyBearElem>,
        groups: &[&IntelBuffer<BabyBearElem>],
        globals: &[&IntelBuffer<BabyBearElem>],
        poly_mix: BabyBearExtElem,
        po2: usize,
        steps: usize,
    ) {
        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        let poly_mix_pows = map_pow(poly_mix, crate::info::POLY_MIX_POWERS);

        // GPU eval_check: -O1 frontend + -cl-opt-disable for ocloc (fast compile).
        // At po2=18 (1M domain), the unoptimized GPU code may run within TDR limits.
        let poly_mix_buf: IntelBuffer<u32> = IntelBuffer::copy_from(
            "poly_mix",
            unsafe {
                std::slice::from_raw_parts(
                    poly_mix_pows.as_ptr() as *const u32,
                    poly_mix_pows.len() * 4,
                )
            },
        );

        let rou = BabyBearElem::ROU_FWD[po2 + EXP_PO2];
        let rou_raw: u32 = unsafe { std::mem::transmute(rou) };

        let queue = risc0_sys::intel::get_queue();

        risc0_sys::intel::esimd_check(unsafe {
            risc0_circuit_recursion_sys::risc0_circuit_recursion_intel_eval_check(
                queue,
                check.as_device_ptr().0 as *mut std::ffi::c_void,
                groups[REGISTER_GROUP_CTRL].as_device_ptr().0 as *const std::ffi::c_void,
                groups[REGISTER_GROUP_DATA].as_device_ptr().0 as *const std::ffi::c_void,
                groups[REGISTER_GROUP_ACCUM].as_device_ptr().0 as *const std::ffi::c_void,
                globals[GLOBAL_MIX].as_device_ptr().0 as *const std::ffi::c_void,
                globals[GLOBAL_OUT].as_device_ptr().0 as *const std::ffi::c_void,
                poly_mix_buf.as_device_ptr().0 as *const std::ffi::c_void,
                rou_raw,
                po2 as u32,
                domain as u32,
            )
        });
    }

    #[allow(unused)]
    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        ctrl: &IntelBuffer<BabyBearElem>,
        io: &IntelBuffer<BabyBearElem>,
        data: &IntelBuffer<BabyBearElem>,
        mix: &IntelBuffer<BabyBearElem>,
        accum: &IntelBuffer<BabyBearElem>,
        steps: usize,
    ) {
        unimplemented!()
    }
}

type IntelCircuitHalPoseidon2 = IntelCircuitHal<IntelHashPoseidon2>;
type IntelCircuitHalSha256 = IntelCircuitHal<IntelHashSha256>;
type IntelCircuitHalPoseidon254 = IntelCircuitHal<IntelHashPoseidon254>;

pub(crate) fn recursion_prover(hashfn: &str) -> Result<Box<dyn RecursionProver>> {
    match hashfn {
        "poseidon2" => {
            let hal = Rc::new(IntelHalPoseidon2::new());
            let circuit_hal = Rc::new(IntelCircuitHalPoseidon2::new(hal.clone()));
            Ok(Box::new(RecursionProverImpl::new(hal, circuit_hal)))
        }
        "sha-256" => {
            let hal = Rc::new(IntelHalSha256::new());
            let circuit_hal = Rc::new(IntelCircuitHalSha256::new(hal.clone()));
            Ok(Box::new(RecursionProverImpl::new(hal, circuit_hal)))
        }
        "poseidon_254" => {
            // GPU Poseidon-254 via ESIMD — BN254 field arithmetic on Intel Arc.
            // hash_fold/hash_rows run entirely on GPU with zero PCIe round-trips.
            let hal = Rc::new(IntelHalPoseidon254::new());
            let circuit_hal = Rc::new(IntelCircuitHalPoseidon254::new(hal.clone()));
            Ok(Box::new(RecursionProverImpl::new(hal, circuit_hal)))
        }
        _ => bail!("Unsupported hashfn: {hashfn}"),
    }
}
