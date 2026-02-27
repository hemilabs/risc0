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

use anyhow::Result;
use risc0_circuit_rv32im_sys::{
    risc0_circuit_rv32im_cuda_accum, risc0_circuit_rv32im_cuda_eval_check,
    risc0_circuit_rv32im_cuda_witgen, RawAccumBuffers, RawBuffer, RawExecBuffers,
    RawPreflightTrace,
};
use risc0_core::{
    field::{map_pow, Elem, ExtElem as _, RootsOfUnity},
    scope,
};
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    core::log2_ceil,
    hal::{
        hip::{BufferImpl as HipBuffer, HipHal, HipHalPoseidon2, HipHash, HipHashPoseidon2},
        AccumPreflight, Buffer, CircuitHal,
    },
    INV_RATE,
};

use crate::{
    prove::{SegmentProver, GLOBAL_MIX, GLOBAL_OUT},
    zirgen::{
        circuit::{ExtVal, Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE, REGISTER_GROUP_DATA},
        info::{NUM_POLY_MIX_POWERS, POLY_MIX_POWERS},
    },
};

use super::{
    CircuitAccumulator, CircuitWitnessGenerator, MetaBuffer, PreflightTrace, SegmentProverImpl,
    StepMode,
};

pub struct HipCircuitHal<HH: HipHash> {
    _hal: Rc<HipHal<HH>>, // retain a reference to ensure the context remains valid
}

impl<HH: HipHash> HipCircuitHal<HH> {
    pub fn new(_hal: Rc<HipHal<HH>>) -> Self {
        Self { _hal }
    }
}

impl<HH: HipHash> CircuitWitnessGenerator<HipHal<HH>> for HipCircuitHal<HH> {
    fn generate_witness(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        global: &MetaBuffer<HipHal<HH>>,
        data: &MetaBuffer<HipHal<HH>>,
    ) -> Result<()> {
        scope!("witgen");

        let cycles = preflight.cycles.len();
        assert_eq!(cycles, data.rows);
        tracing::debug!("witgen: {cycles}");

        let global_ptr = global.buf.as_device_ptr();
        let data_ptr = data.buf.as_device_ptr();
        let buffers = RawExecBuffers {
            global: RawBuffer {
                buf: global_ptr.as_ptr() as *const Val,
                rows: global.rows,
                cols: global.cols,
                checked: global.checked,
            },
            data: RawBuffer {
                buf: data_ptr.as_ptr() as *const Val,
                rows: data.rows,
                cols: data.cols,
                checked: data.checked,
            },
        };

        let preflight = RawPreflightTrace {
            cycles: preflight.cycles.as_ptr(),
            txns: preflight.txns.as_ptr(),
            bigint_bytes: preflight.bigint_bytes.as_ptr(),
            txns_len: preflight.txns.len() as u32,
            bigint_bytes_len: preflight.bigint_bytes.len() as u32,
            table_split_cycle: preflight.table_split_cycle,
        };
        ffi_wrap(|| unsafe {
            risc0_circuit_rv32im_cuda_witgen(mode as u32, &buffers, &preflight, cycles as u32)
        })
    }
}

impl<HH: HipHash> CircuitAccumulator<HipHal<HH>> for HipCircuitHal<HH> {
    fn step_accum(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<HipHal<HH>>,
        accum: &MetaBuffer<HipHal<HH>>,
        global: &MetaBuffer<HipHal<HH>>,
        mix: &MetaBuffer<HipHal<HH>>,
    ) -> Result<()> {
        scope!("accumulate");

        let cycles = preflight.cycles.len();
        tracing::debug!("accumulate: {cycles}");

        let buffers = RawAccumBuffers {
            data: RawBuffer {
                buf: data.buf.as_device_ptr().as_ptr() as *const Val,
                rows: data.rows,
                cols: data.cols,
                checked: data.checked,
            },
            accum: RawBuffer {
                buf: accum.buf.as_device_ptr().as_ptr() as *const Val,
                rows: accum.rows,
                cols: accum.cols,
                // Disable checked reads/writes so that in-place
                // changes can be made during phase2 and phase3 of accumulation.
                checked: false,
            },
            global: RawBuffer {
                buf: global.buf.as_device_ptr().as_ptr() as *const Val,
                rows: global.rows,
                cols: global.cols,
                checked: global.checked,
            },
            mix: RawBuffer {
                buf: mix.buf.as_device_ptr().as_ptr() as *const Val,
                rows: mix.rows,
                cols: mix.cols,
                checked: mix.checked,
            },
        };
        let preflight = RawPreflightTrace {
            cycles: preflight.cycles.as_ptr(),
            txns: preflight.txns.as_ptr(),
            bigint_bytes: preflight.bigint_bytes.as_ptr(),
            txns_len: preflight.txns.len() as u32,
            bigint_bytes_len: preflight.bigint_bytes.len() as u32,
            table_split_cycle: preflight.table_split_cycle,
        };
        ffi_wrap(|| unsafe { risc0_circuit_rv32im_cuda_accum(&buffers, &preflight, cycles as u32) })
    }
}

impl<HH: HipHash> CircuitHal<HipHal<HH>> for HipCircuitHal<HH> {
    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        _ctrl: &HipBuffer<Val>,
        _io: &HipBuffer<Val>,
        _data: &HipBuffer<Val>,
        _mix: &HipBuffer<Val>,
        _accum: &HipBuffer<Val>,
        _steps: usize,
    ) {
    }

    fn eval_check(
        &self,
        check: &HipBuffer<Val>,
        groups: &[&HipBuffer<Val>],
        globals: &[&HipBuffer<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) {
        scope!("eval_check");

        let accum = groups[REGISTER_GROUP_ACCUM];
        let ctrl = groups[REGISTER_GROUP_CODE];
        let data = groups[REGISTER_GROUP_DATA];
        let mix = globals[GLOBAL_MIX];
        let out = globals[GLOBAL_OUT];
        tracing::debug!(
            "check: {}, ctrl: {}, data: {}, accum: {}, mix: {} out: {}",
            check.size(),
            ctrl.size(),
            data.size(),
            accum.size(),
            mix.size(),
            out.size()
        );
        tracing::debug!(
            "total: {}",
            (check.size() + ctrl.size() + data.size() + accum.size() + mix.size() + out.size()) * 4
        );

        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        let rou = Val::ROU_FWD[po2 + EXP_PO2];

        tracing::debug!("steps: {steps}, domain: {domain}, po2: {po2}, rou: {rou:?}");
        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);
        let poly_mix_pows: &[u32; ExtVal::EXT_SIZE * NUM_POLY_MIX_POWERS] =
            ExtVal::as_u32_slice(poly_mix_pows.as_slice())
                .try_into()
                .unwrap();

        ffi_wrap(|| unsafe {
            risc0_circuit_rv32im_cuda_eval_check(
                check.as_device_ptr(),
                ctrl.as_device_ptr(),
                data.as_device_ptr(),
                accum.as_device_ptr(),
                mix.as_device_ptr(),
                out.as_device_ptr(),
                &rou as *const Val,
                po2 as u32,
                domain as u32,
                poly_mix_pows.as_ptr(),
            )
        })
        .unwrap();
    }
}

pub type HipCircuitHalPoseidon2 = HipCircuitHal<HipHashPoseidon2>;

pub fn segment_prover() -> Result<Box<dyn SegmentProver>> {
    let hal_factory = || {
        let hal = Rc::new(HipHalPoseidon2::new());
        let circuit_hal = Rc::new(HipCircuitHalPoseidon2::new(hal.clone()));
        (hal, circuit_hal)
    };
    Ok(Box::new(SegmentProverImpl::new(hal_factory)))
}
