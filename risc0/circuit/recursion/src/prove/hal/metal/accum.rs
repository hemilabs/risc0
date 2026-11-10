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

//! Recursion accumulation on the Metal HAL.
//!
//! C5 placeholder: runs the CPU accum (`risc0_circuit_recursion_cpu_accum`, as `hal/cpu.rs` calls
//! it) over host views of the Metal HAL's shared buffers, after a `sync()`. C7 replaces this with
//! one command buffer (fillOneExt, parStepComputeAccum, prefix trio, parStepVerifyAccum).

use anyhow::Result;
use risc0_circuit_recursion_sys::{risc0_circuit_recursion_cpu_accum, RawAccumBuffers};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    field::baby_bear::BabyBearElem,
    hal::metal::{BufferImpl, MetalHal, MetalHash},
};

use super::{host_slice, MetalCircuitHal, Phase};
use crate::prove::hal::CircuitAccumulator;

impl<MH: MetalHash> CircuitAccumulator<MetalHal<MH>> for MetalCircuitHal<MH> {
    fn accumulate(
        &self,
        work_cycles: u32,
        total_cycles: u32,
        ctrl: &BufferImpl<BabyBearElem>,
        global: &BufferImpl<BabyBearElem>,
        data: &BufferImpl<BabyBearElem>,
        mix: &BufferImpl<BabyBearElem>,
        accum: &BufferImpl<BabyBearElem>,
    ) -> Result<()> {
        scope!("metal_accum(cpu)");
        self.enter(Phase::Accum);
        self.sync()?;
        // SAFETY: synced, and nothing is encoded until this call returns.
        let (ctrl, global, data, mix, accum) = unsafe {
            (
                host_slice(ctrl),
                host_slice(global),
                host_slice(data),
                host_slice(mix),
                host_slice(accum),
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
            risc0_circuit_recursion_cpu_accum(&buffers, work_cycles, total_cycles)
        })
    }
}
