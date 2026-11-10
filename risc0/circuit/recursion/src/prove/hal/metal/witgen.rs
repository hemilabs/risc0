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

//! Recursion witness generation on the Metal HAL.
//!
//! C5 placeholder: runs the CPU witgen (`risc0_circuit_recursion_cpu_witgen`, as `hal/cpu.rs`
//! calls it) over host views of the Metal HAL's shared buffers, after a `sync()`. Host chains are
//! ignored (the CPU witgen runs every chain), as for the CPU HAL. C6 replaces this with the Metal
//! Parallel path (parStepExec, CPU sort/scan, injectWomBacks, parStepVerifyWom).

use anyhow::Result;
use risc0_circuit_recursion_sys::{
    risc0_circuit_recursion_cpu_witgen, RawExecBuffers, RawPreflightCycle, RawPreflightTrace,
    StepMode,
};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    field::baby_bear::BabyBearElem,
    hal::metal::{BufferImpl, MetalHal, MetalHash},
};

use super::{grow, host_slice, MetalCircuitHal, Phase};
use crate::prove::hal::CircuitWitnessGenerator;

const CYCLE_WORDS: usize = std::mem::size_of::<RawPreflightCycle>() / 4;
const _: () = assert!(std::mem::size_of::<RawPreflightCycle>() == 8);

impl<MH: MetalHash> CircuitWitnessGenerator<MetalHal<MH>> for MetalCircuitHal<MH> {
    fn generate_witness(
        &self,
        mode: StepMode,
        total_cycles: u32,
        preflight: &RawPreflightTrace,
        ctrl: &BufferImpl<BabyBearElem>,
        data: &BufferImpl<BabyBearElem>,
        global: &BufferImpl<BabyBearElem>,
    ) -> Result<()> {
        scope!("metal_witgen(cpu)");
        self.enter(Phase::Witgen);
        let num_cycles = preflight.num_cycles as usize;
        tracing::debug!(
            "witgen: {num_cycles} cycles of {total_cycles} (CPU kernels over Metal buffers)"
        );
        // The witgen advances `cycles[c].iopIdx`: work on a HAL-owned copy (PLAN 2.6) so the
        // caller's trace stays pristine and the witgen can be re-run on it (prewitgen and host
        // chain checks).
        let cycles = grow(
            &self.hal,
            &mut self.scratch().cycles,
            "rec_cycles",
            (num_cycles * CYCLE_WORDS).max(1),
        );
        self.record(Phase::Witgen);
        self.sync()?;
        // SAFETY: synced, nothing is encoded until this call returns; `cycles` holds
        // `num_cycles` RawPreflightCycle (2 x u32, size asserted) and the source as many.
        unsafe {
            if num_cycles > 0 {
                std::ptr::copy_nonoverlapping(
                    preflight.cycles as *const u32,
                    cycles.as_mut_ptr(),
                    num_cycles * CYCLE_WORDS,
                );
            }
        }
        let (ctrl, data, global) =
            unsafe { (host_slice(ctrl), host_slice(data), host_slice(global)) };
        let buffers = RawExecBuffers {
            ctrl: ctrl.as_ptr(),
            data: data.as_ptr(),
            global: global.as_ptr(),
        };
        let trace = RawPreflightTrace {
            wom: preflight.wom,
            cycles: cycles.as_ptr() as *const RawPreflightCycle,
            iops: preflight.iops,
            num_woms: preflight.num_woms,
            num_cycles: preflight.num_cycles,
            num_iops: preflight.num_iops,
        };
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_cpu_witgen(mode, &buffers, &trace, total_cycles)
        })
    }
}
