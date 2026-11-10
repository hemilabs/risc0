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

//! Witness generation of the keccak circuit on the Metal HAL (milestone 1, item K2).
//!
//! - `scatter_preflight` runs the `scatter_preflight` kernel (kernels/metal/scatter.metal, a 1:1
//!   port of kernels/cuda/ffi.cu `scatter_preflight`), as `CudaCircuitHal::scatter_preflight`
//!   does, into the `data` buffer the prover allocated (INVALID-filled).
//! - `generate_witness` is the CPU witness generator (`generate_witness_host`) writing straight
//!   into the HAL's shared (unified-memory) `global`/`data` buffers through `view_mut`, which
//!   first waits for the scatter's batch. This is the "host witness" of PLAN 4.4 phase 1
//!   (`HOST_WITNESS = true`, as on Intel), so a witness `host_witness` made can be proved too.
//!   K6 replaces it with the per-arm GPU kernels.

use anyhow::{bail, ensure, Result};
use risc0_circuit_keccak_sys::ScatterInfo;
use risc0_core::scope;
use risc0_zkp::hal::{
    metal::{as_bytes, BufferImpl as MetalBuffer, Grid, KernelArg, MetalHal, MetalHash},
    Buffer as _,
};

use super::{
    super::{
        cpu::{generate_witness_host, HostMetaBuffer},
        CircuitWitnessGenerator, ForwardPreflightOrder, MetaBuffer, PreflightCycleOrder,
        PreflightTrace, StepMode,
    },
    MetalCircuitHal,
};

/// `scatter_preflight` (kernels/metal/scatter.metal) bindings.
const SCATTER_INTO: usize = 0;
const SCATTER_INFOS: usize = 1;
const SCATTER_FROM: usize = 2;
const SCATTER_FAULT: usize = 3;
const SCATTER_PARAMS: usize = 4;
const SCATTER_NARGS: usize = 5;

/// `ScatterParams` of scatter.metal: `rows`, `cols`, `count`, `from_len` (16 bytes, set_bytes).
type ScatterParams = [u32; 4];
const _: () = assert!(std::mem::size_of::<ScatterParams>() == 16);

impl<MH: MetalHash> MetalCircuitHal<MH> {
    /// Scatter the preflight's packed words `from` into `into` on the GPU: the CUDA
    /// `scatter_preflight` kernel, a thread per info. Syncs, and fails if the kernel rejected an
    /// info (out of bounds, or bits not in 1..32), where the host path would panic.
    pub(crate) fn scatter_preflight_gpu(
        &self,
        into: &MetaBuffer<MetalHal<MH>>,
        infos: &[ScatterInfo],
        from: &[u32],
    ) -> Result<()> {
        ensure!(into.buf.size() == into.rows * into.cols);
        ensure!(
            u32::try_from(into.buf.size()).is_ok(),
            "scatter: witness of {} elements does not fit 32-bit indexing",
            into.buf.size()
        );
        if infos.is_empty() {
            return Ok(());
        }
        let infos_buf: MetalBuffer<ScatterInfo> = self.hal.copy_from_slice("scatter_infos", infos);
        // A zero-length Metal buffer cannot be made; a lone 0 is never read (count == 0 infos).
        let from_buf = if from.is_empty() {
            self.hal.copy_from_slice("scatter_from", &[0u32])
        } else {
            self.hal.copy_from_slice("scatter_from", from)
        };
        let fault = self.hal.copy_from_slice("scatter_fault", &[0u32]);
        let params: ScatterParams = [
            into.rows as u32,
            into.cols as u32,
            u32::try_from(infos.len())?,
            u32::try_from(from.len())?,
        ];

        let mut args: [KernelArg; SCATTER_NARGS] = std::array::from_fn(|_| KernelArg::Null);
        args[SCATTER_INTO] = into.buf.as_arg();
        args[SCATTER_INFOS] = infos_buf.as_arg();
        args[SCATTER_FROM] = from_buf.as_arg();
        args[SCATTER_FAULT] = fault.as_arg();
        args[SCATTER_PARAMS] = KernelArg::Bytes(as_bytes(&params));
        self.hal.dispatch(
            self.pipeline("scatter_preflight"),
            &args,
            Grid::Threads(infos.len() as u64),
            &[],
        );
        // End of the scatter phase (plan 2.6): the fault flag is read right after.
        self.hal.try_sync()?;
        if fault.get_at(0) != 0 {
            bail!(
                "keccak scatter_preflight (Metal): the preflight trace has an invalid ScatterInfo"
            );
        }
        Ok(())
    }
}

impl<MH: MetalHash> CircuitWitnessGenerator<MetalHal<MH>> for MetalCircuitHal<MH> {
    type PreferredPreflightOrder = ForwardPreflightOrder;
    // The witness is the CPU's (`generate_witness` below), so a `host_witness` can be proved.
    const HOST_WITNESS: bool = true;

    fn scatter_preflight(
        &self,
        into: &MetaBuffer<MetalHal<MH>>,
        infos: &[ScatterInfo],
        from: &[u32],
    ) -> Result<()> {
        scope!("scatter");
        #[cfg(test)]
        super::tests::mark_phase(&self.hal.device, super::tests::Phase::Scatter);
        self.scatter_preflight_gpu(into, infos, from)
    }

    fn generate_witness<O: PreflightCycleOrder>(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace<O>,
        global: &MetaBuffer<MetalHal<MH>>,
        data: &MetaBuffer<MetalHal<MH>>,
    ) -> Result<()> {
        scope!("witgen");
        #[cfg(test)]
        super::tests::mark_phase(&self.hal.device, super::tests::Phase::Witgen);
        // The CPU witness generator, writing straight into the shared buffers (unified memory);
        // `view_mut` waits for any open batch (the scatter) first.
        let mut result = Ok(());
        global.buf.view_mut(|global_host| {
            data.buf.view_mut(|data_host| {
                result = generate_witness_host(
                    mode,
                    preflight,
                    HostMetaBuffer::of(global, global_host),
                    HostMetaBuffer::of(data, data_host),
                );
            })
        });
        #[cfg(test)]
        super::tests::mark_phase(&self.hal.device, super::tests::Phase::Commit);
        result
    }
}
