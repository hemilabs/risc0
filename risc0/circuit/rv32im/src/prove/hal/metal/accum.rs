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

//! rv32im accumulation on the Metal HAL.
//!
//! R7 placeholder: delegates to the CPU accum kernel (`risc0_circuit_rv32im_cpu_accum`, as
//! `hal/cpu.rs` calls it) over host views of the Metal HAL's shared buffers. R9 replaces this with
//! the Metal accum path (`ctx.require(Phase::Accum)`, CPU perm, step_accum, scans, finalize).

use anyhow::Result;
use risc0_circuit_rv32im_sys::{
    risc0_circuit_rv32im_cpu_accum, RawAccumBuffers, RawBuffer, RawPreflightTrace,
};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::hal::metal::{MetalHal, MetalHash};

use super::{count, host_slice, MetalCircuitHal, Phase};
use crate::prove::hal::{CircuitAccumulator, MetaBuffer, PreflightTrace};

impl<MH: MetalHash> CircuitAccumulator<MetalHal<MH>> for MetalCircuitHal<MH> {
    fn step_accum(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<MetalHal<MH>>,
        accum: &MetaBuffer<MetalHal<MH>>,
        global: &MetaBuffer<MetalHal<MH>>,
        mix: &MetaBuffer<MetalHal<MH>>,
    ) -> Result<()> {
        scope!("metal_accumulate(cpu)");
        count(Phase::Accum);
        let cycles = preflight.cycles.len();
        tracing::debug!("accumulate: {cycles} (CPU kernels over Metal buffers)");
        self.sync()?;
        let raw = |m: &MetaBuffer<MetalHal<MH>>| RawBuffer {
            // SAFETY: synced, and nothing is encoded until this call returns.
            buf: unsafe { host_slice(&m.buf) }.as_ptr(),
            rows: m.rows,
            cols: m.cols,
            checked: m.checked,
        };
        let buffers = RawAccumBuffers {
            data: raw(data),
            accum: raw(accum),
            global: raw(global),
            mix: raw(mix),
        };
        let preflight = RawPreflightTrace {
            cycles: preflight.cycles.as_ptr(),
            txns: preflight.txns.as_ptr(),
            bigint_bytes: preflight.bigint_bytes.as_ptr(),
            txns_len: preflight.txns.len() as u32,
            bigint_bytes_len: preflight.bigint_bytes.len() as u32,
            table_split_cycle: preflight.table_split_cycle,
        };
        ffi_wrap(|| unsafe { risc0_circuit_rv32im_cpu_accum(&buffers, &preflight, cycles as u32) })
    }

    // step_accum_zeroizes: false (the CPU kernel leaves INVALID cells; the prover zeroizes).
}
