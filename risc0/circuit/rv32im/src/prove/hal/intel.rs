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

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Result;
// CPU accum FFI used as fallback while GPU accum is being debugged
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    core::log2_ceil,
    field::{map_pow, RootsOfUnity as _},
    hal::{
        intel::{
            BufferImpl as IntelBuffer, IntelHal, IntelHalPoseidon2, IntelHash, IntelHashPoseidon2,
        },
        AccumPreflight, CircuitHal, Hal,
    },
    INV_RATE,
};

use super::{
    CircuitAccumulator, CircuitWitnessGenerator, MetaBuffer, SegmentProver, SegmentProverImpl,
    StepMode,
};
use crate::{
    prove::{witgen::preflight::PreflightTrace, GLOBAL_MIX, GLOBAL_OUT},
    zirgen::{
        circuit::{ExtVal, Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_DATA},
        info::POLY_MIX_POWERS,
    },
};

pub struct IntelCircuitHal<IH: IntelHash> {
    _hal: Rc<IntelHal<IH>>,
    // Keep buffers alive while eval_check runs asynchronously on separate queue.
    eval_check_poly_mix: RefCell<Option<IntelBuffer<u32>>>,
    eval_check_inter_fp: RefCell<Option<IntelBuffer<u32>>>,
    eval_check_inter_ext: RefCell<Option<IntelBuffer<u32>>>,
}

impl<IH: IntelHash> IntelCircuitHal<IH> {
    pub fn new(_hal: Rc<IntelHal<IH>>) -> Self {
        Self {
            _hal,
            eval_check_poly_mix: RefCell::new(None),
            eval_check_inter_fp: RefCell::new(None),
            eval_check_inter_ext: RefCell::new(None),
        }
    }
}

impl<IH: IntelHash> CircuitWitnessGenerator<IntelHal<IH>> for IntelCircuitHal<IH> {
    fn generate_witness(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        global: &MetaBuffer<IntelHal<IH>>,
        data: &MetaBuffer<IntelHal<IH>>,
        pre_data: &MetaBuffer<IntelHal<IH>>,
    ) -> Result<()> {
        scope!("intel_witgen");
        let cycles = preflight.cycles.len();
        tracing::debug!("witgen: {cycles} cycles (GPU)");

        let queue = risc0_sys::intel::get_queue();
        ffi_wrap(|| unsafe {
            risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_witgen(
                queue,
                mode as u32,
                data.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                data.rows as u32,
                data.cols as u32,
                pre_data.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                global.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                global.cols as u32,
                preflight.cycles.as_ptr(),
                preflight.cycles.len() as u32,
                preflight.txns.as_ptr(),
                preflight.txns.len() as u32,
                preflight.bigint_bytes.as_ptr(),
                preflight.bigint_bytes.len() as u32,
                preflight.table_split_cycle,
                cycles as u32,
            )
        })?;

        Ok(())
    }
}

impl<IH: IntelHash> CircuitAccumulator<IntelHal<IH>> for IntelCircuitHal<IH> {
    fn step_accum(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<IntelHal<IH>>,
        accum: &MetaBuffer<IntelHal<IH>>,
        global: &MetaBuffer<IntelHal<IH>>,
        mix: &MetaBuffer<IntelHal<IH>>,
    ) -> Result<()> {
        scope!("intel_accumulate");
        let cycles = preflight.cycles.len();

        // GPU accum compiled at -Os (testing if different opts avoid icpx -O1 miscompilation)
        tracing::debug!("accumulate: {cycles} cycles (GPU, -Os)");
        let queue = risc0_sys::intel::get_queue();
        ffi_wrap(|| unsafe {
            risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_accum(
                queue,
                data.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                data.rows as u32, data.cols as u32,
                accum.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                accum.rows as u32, accum.cols as u32,
                global.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                global.cols as u32,
                mix.buf.as_device_ptr().0 as *mut std::ffi::c_void,
                mix.cols as u32,
                preflight.cycles.as_ptr(), preflight.cycles.len() as u32,
                preflight.txns.as_ptr(), preflight.txns.len() as u32,
                preflight.bigint_bytes.as_ptr(), preflight.bigint_bytes.len() as u32,
                preflight.table_split_cycle, cycles as u32,
            )
        })?;

        Ok(())
    }
}

impl<IH: IntelHash> CircuitHal<IntelHal<IH>> for IntelCircuitHal<IH> {
    fn eval_check(
        &self,
        check: &IntelBuffer<Val>,
        groups: &[&IntelBuffer<Val>],
        globals: &[&IntelBuffer<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) {
        scope!("eval_check");

        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);

        // Upload poly_mix_pows to GPU. Store in struct to keep alive while
        // eval_check runs asynchronously on separate queue.
        let poly_mix_buf: IntelBuffer<u32> = IntelBuffer::copy_from(
            "poly_mix",
            unsafe {
                std::slice::from_raw_parts(
                    poly_mix_pows.as_ptr() as *const u32,
                    poly_mix_pows.len() * 4,
                )
            },
        );

        let rou = Val::ROU_FWD[po2 + EXP_PO2];
        let rou_raw: u32 = unsafe { std::mem::transmute(rou) };

        let eval_queue = risc0_sys::intel::get_eval_check_queue();

        let use_multipass = std::env::var_os("RISC0_MULTIPASS").is_some();

        if use_multipass {
            // 2-way multi-pass: allocate intermediate buffer, run pass1 then pass2
            // Intermediate: 197 Fp (u32) + 9 FpExt (7 by-val + 2 x34 cross-boundary)
            let n_inter_fp = 197 * domain;
            let n_inter_ext = 9 * domain * 4;
            let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
            let t0 = if verbose { Some(std::time::Instant::now()) } else { None };

            let inter_fp = self._hal.alloc_u32("inter_fp", n_inter_fp);
            let inter_ext = self._hal.alloc_u32("inter_ext", n_inter_ext);

            // Pass 1: upper chain → writes intermediate
            risc0_sys::intel::esimd_check(unsafe {
                risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_eval_check_pass1(
                    eval_queue,
                    inter_fp.as_device_ptr().0 as *mut std::ffi::c_void,
                    inter_ext.as_device_ptr().0 as *mut std::ffi::c_void,
                    groups[REGISTER_GROUP_DATA].as_device_ptr().0 as *const std::ffi::c_void,
                    groups[REGISTER_GROUP_ACCUM].as_device_ptr().0 as *const std::ffi::c_void,
                    globals[GLOBAL_OUT].as_device_ptr().0 as *const std::ffi::c_void,
                    globals[GLOBAL_MIX].as_device_ptr().0 as *const std::ffi::c_void,
                    poly_mix_buf.as_device_ptr().0 as *const std::ffi::c_void,
                    domain as u32,
                )
            });

            if verbose {
                // Sync to measure pass1 time
                risc0_sys::intel::esimd_check(unsafe {
                    risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_eval_check_sync(eval_queue)
                });
                let t1 = std::time::Instant::now();
                eprintln!("      [eval_check_pass1] {:.1}ms",
                    t1.duration_since(t0.unwrap()).as_secs_f64() * 1000.0);
            }

            let t1 = if verbose { Some(std::time::Instant::now()) } else { None };

            // Pass 2: reads intermediate → writes check buffer
            risc0_sys::intel::esimd_check(unsafe {
                risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_eval_check_pass2(
                    eval_queue,
                    check.as_device_ptr().0 as *mut std::ffi::c_void,
                    inter_fp.as_device_ptr().0 as *const std::ffi::c_void,
                    inter_ext.as_device_ptr().0 as *const std::ffi::c_void,
                    groups[REGISTER_GROUP_DATA].as_device_ptr().0 as *const std::ffi::c_void,
                    groups[REGISTER_GROUP_ACCUM].as_device_ptr().0 as *const std::ffi::c_void,
                    globals[GLOBAL_OUT].as_device_ptr().0 as *const std::ffi::c_void,
                    globals[GLOBAL_MIX].as_device_ptr().0 as *const std::ffi::c_void,
                    poly_mix_buf.as_device_ptr().0 as *const std::ffi::c_void,
                    rou_raw,
                    po2 as u32,
                    domain as u32,
                )
            });

            if verbose {
                risc0_sys::intel::esimd_check(unsafe {
                    risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_eval_check_sync(eval_queue)
                });
                let t2 = std::time::Instant::now();
                eprintln!("      [eval_check_pass2] {:.1}ms",
                    t2.duration_since(t1.unwrap()).as_secs_f64() * 1000.0);
            }

            // Keep intermediate buffers alive until eval_check_dep()
            *self.eval_check_inter_fp.borrow_mut() = Some(inter_fp);
            *self.eval_check_inter_ext.borrow_mut() = Some(inter_ext);
        } else {
            // Monolithic: single kernel call
            risc0_sys::intel::esimd_check(unsafe {
                risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_eval_check(
                    eval_queue,
                    check.as_device_ptr().0 as *mut std::ffi::c_void,
                    groups[REGISTER_GROUP_DATA].as_device_ptr().0 as *const std::ffi::c_void,
                    groups[REGISTER_GROUP_ACCUM].as_device_ptr().0 as *const std::ffi::c_void,
                    globals[GLOBAL_OUT].as_device_ptr().0 as *const std::ffi::c_void,
                    globals[GLOBAL_MIX].as_device_ptr().0 as *const std::ffi::c_void,
                    poly_mix_buf.as_device_ptr().0 as *const std::ffi::c_void,
                    rou_raw,
                    po2 as u32,
                    domain as u32,
                )
            });
        }

        // Keep poly_mix_buf alive until eval_check_dep() or next eval_check()
        *self.eval_check_poly_mix.borrow_mut() = Some(poly_mix_buf);
    }

    fn eval_check_dep(&self) {
        // Wait for eval_check (running on separate queue) before the main
        // queue proceeds with operations that depend on the check buffer.
        let eval_queue = risc0_sys::intel::get_eval_check_queue();
        risc0_sys::intel::esimd_check(unsafe {
            risc0_circuit_rv32im_sys::risc0_circuit_rv32im_intel_eval_check_sync(eval_queue)
        });
        // Release all eval_check buffers now that it's done
        *self.eval_check_poly_mix.borrow_mut() = None;
        *self.eval_check_inter_fp.borrow_mut() = None;
        *self.eval_check_inter_ext.borrow_mut() = None;
    }

    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        _ctrl: &IntelBuffer<Val>,
        _global: &IntelBuffer<Val>,
        _data: &IntelBuffer<Val>,
        _mix: &IntelBuffer<Val>,
        _accum: &IntelBuffer<Val>,
        _steps: usize,
    ) {
        unimplemented!()
    }
}

pub type IntelCircuitHalPoseidon2 = IntelCircuitHal<IntelHashPoseidon2>;

pub fn segment_prover() -> Result<Box<dyn SegmentProver>> {
    let hal_factory = || {
        let hal = Rc::new(IntelHalPoseidon2::new());
        let circuit_hal = Rc::new(IntelCircuitHalPoseidon2::new(hal.clone()));
        (hal, circuit_hal)
    };
    Ok(Box::new(SegmentProverImpl::new(hal_factory)))
}
