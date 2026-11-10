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

//! rv32im witness generation on the Metal HAL.
//!
//! R7 placeholder: delegates to the CPU witgen kernel (`risc0_circuit_rv32im_cpu_witgen`, as
//! `hal/cpu.rs` calls it) over host views of the Metal HAL's shared buffers. R8 replaces this with
//! the Metal witgen path (`ctx.require(Phase::Witgen)`, one command buffer).

use anyhow::Result;
use risc0_circuit_rv32im_sys::{
    risc0_circuit_rv32im_cpu_witgen, RawBuffer, RawExecBuffers, RawPreflightTrace,
};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::hal::metal::{MetalHal, MetalHash};

use super::{count, host_slice, MetalCircuitHal, Phase};
use crate::prove::hal::{CircuitWitnessGenerator, MetaBuffer, PreflightTrace, StepMode};

impl<MH: MetalHash> CircuitWitnessGenerator<MetalHal<MH>> for MetalCircuitHal<MH> {
    // writes_cycle_cols / takes_zeroed_data: false, as for the CPU circuit HAL (its checked sets
    // expect INVALID-filled data, and it does not write the deferred cycle columns).

    fn generate_witness(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        global: &MetaBuffer<MetalHal<MH>>,
        data: &MetaBuffer<MetalHal<MH>>,
        _pre_data: &MetaBuffer<MetalHal<MH>>,
    ) -> Result<()> {
        scope!("metal_witgen(cpu)");
        count(Phase::Witgen);
        let cycles = preflight.cycles.len();
        tracing::debug!("witgen: {cycles} (CPU kernels over Metal buffers)");
        self.sync()?;
        // SAFETY: synced, and nothing is encoded until this call returns.
        let (global_buf, data_buf) = unsafe { (host_slice(&global.buf), host_slice(&data.buf)) };
        let buffers = RawExecBuffers {
            global: RawBuffer {
                buf: global_buf.as_ptr(),
                rows: global.rows,
                cols: global.cols,
                checked: global.checked,
            },
            data: RawBuffer {
                buf: data_buf.as_ptr(),
                rows: data.rows,
                cols: data.cols,
                checked: data.checked,
            },
            // CPU runs sequentially; no race condition, so pre_data aliases data (as hal/cpu.rs).
            pre_data: RawBuffer {
                buf: data_buf.as_ptr(),
                rows: data.rows,
                cols: data.cols,
                checked: false,
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
            risc0_circuit_rv32im_cpu_witgen(mode as u32, &buffers, &preflight, cycles as u32)
        })
    }
}
