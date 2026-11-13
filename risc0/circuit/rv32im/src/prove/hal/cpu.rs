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

use anyhow::Result;
use rayon::prelude::*;
use risc0_circuit_rv32im_sys::{
    risc0_circuit_rv32im_cpu_accum, risc0_circuit_rv32im_cpu_poly_fp,
    risc0_circuit_rv32im_cpu_witgen, RawAccumBuffers, RawBuffer, RawExecBuffers, RawPreflightTrace,
};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    core::{hash::poseidon2::Poseidon2HashSuite, log2_ceil},
    field::{map_pow, Elem, ExtElem as _, RootsOfUnity as _},
    hal::{cpu::CpuBuffer, AccumPreflight, CircuitHal},
    INV_RATE,
};

use super::{
    CircuitAccumulator, CircuitWitnessGenerator, MetaBuffer, SegmentProver, SegmentProverImpl,
    StepMode,
};
use crate::{
    prove::{witgen::preflight::PreflightTrace, GLOBAL_MIX, GLOBAL_OUT},
    zirgen::{
        circuit::{CircuitField, ExtVal, Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_DATA},
        info::POLY_MIX_POWERS,
    },
};

type CpuHal = risc0_zkp::hal::cpu::CpuHal<CircuitField>;

/// Cycles per call of the 16-wide constraint polynomial.
const POLY_FP_LANES: usize = 16;

type PolyFpX16 = unsafe extern "C" fn(
    usize,
    usize,
    *const ExtVal,
    *const *const Val,
    *mut ExtVal,
) -> *const std::os::raw::c_char;

/// The 16-wide constraint polynomial for this CPU (AVX-512, else AVX2), if the build has it.
/// `RISC0_CPU_EVAL_SCALAR` set forces the scalar one.
fn poly_fp_x16() -> Option<PolyFpX16> {
    #[cfg(target_arch = "x86_64")]
    {
        use risc0_circuit_rv32im_sys::{
            risc0_circuit_rv32im_cpu_poly_fp_x16_available,
            risc0_circuit_rv32im_cpu_poly_fp_x16_avx2, risc0_circuit_rv32im_cpu_poly_fp_x16_avx512,
        };
        if std::env::var_os("RISC0_CPU_EVAL_SCALAR").is_some()
            || !unsafe { risc0_circuit_rv32im_cpu_poly_fp_x16_available() }
        {
            return None;
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            return Some(risc0_circuit_rv32im_cpu_poly_fp_x16_avx512);
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return Some(risc0_circuit_rv32im_cpu_poly_fp_x16_avx2);
        }
    }
    None
}

#[derive(Default)]
pub struct CpuCircuitHal;

impl CircuitWitnessGenerator<CpuHal> for CpuCircuitHal {
    fn generate_witness(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        global: &MetaBuffer<CpuHal>,
        data: &MetaBuffer<CpuHal>,
        _pre_data: &MetaBuffer<CpuHal>,
    ) -> Result<()> {
        scope!("cpu_witgen");
        let cycles = preflight.cycles.len();
        tracing::debug!("witgen: {cycles}");
        let global_buf = global.buf.as_slice();
        let data_buf = data.buf.as_slice();
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
            // CPU runs sequentially; no race condition, so pre_data aliases data.
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

impl CircuitAccumulator<CpuHal> for CpuCircuitHal {
    fn step_accum(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<CpuHal>,
        accum: &MetaBuffer<CpuHal>,
        global: &MetaBuffer<CpuHal>,
        mix: &MetaBuffer<CpuHal>,
    ) -> Result<()> {
        scope!("accumulate");
        let cycles = preflight.cycles.len();
        tracing::debug!("accumulate: {cycles}");
        let data_buf = data.buf.as_slice();
        let accum_buf = accum.buf.as_slice();
        let global_buf = global.buf.as_slice();
        let mix_buf = mix.buf.as_slice();
        let buffers = RawAccumBuffers {
            data: RawBuffer {
                buf: data_buf.as_ptr(),
                rows: data.rows,
                cols: data.cols,
                checked: data.checked,
            },
            accum: RawBuffer {
                buf: accum_buf.as_ptr(),
                rows: accum.rows,
                cols: accum.cols,
                checked: accum.checked,
            },
            global: RawBuffer {
                buf: global_buf.as_ptr(),
                rows: global.rows,
                cols: global.cols,
                checked: global.checked,
            },
            mix: RawBuffer {
                buf: mix_buf.as_ptr(),
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
        ffi_wrap(|| unsafe { risc0_circuit_rv32im_cpu_accum(&buffers, &preflight, cycles as u32) })
    }
}

impl CircuitHal<CpuHal> for CpuCircuitHal {
    fn eval_check(
        &self,
        check: &CpuBuffer<Val>,
        groups: &[&CpuBuffer<Val>],
        globals: &[&CpuBuffer<Val>],
        poly_mix: ExtVal,
        po2: usize,
        steps: usize,
    ) {
        scope!("eval_check");

        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);

        // SAFETY: Convert a borrow of a cell into a raw const slice so that we can pass
        // it over the thread boundary. This should be safe because the scope of the
        // usage is within this function and each thread access will not overlap with
        // each other.

        let data = groups[REGISTER_GROUP_DATA].as_slice();
        let accum = groups[REGISTER_GROUP_ACCUM].as_slice();
        let mix = globals[GLOBAL_MIX].as_slice();
        let out = globals[GLOBAL_OUT].as_slice();
        let check = check.as_slice();

        let data = unsafe { std::slice::from_raw_parts(data.as_ptr(), data.len()) };
        let accum = unsafe { std::slice::from_raw_parts(accum.as_ptr(), accum.len()) };
        let mix = unsafe { std::slice::from_raw_parts(mix.as_ptr(), mix.len()) };
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr(), out.len()) };
        let check = unsafe { std::slice::from_raw_parts(check.as_ptr(), check.len()) };
        let poly_mix_pows = poly_mix_pows.as_slice();

        let args: [*const Val; 4] = [accum.as_ptr(), data.as_ptr(), out.as_ptr(), mix.as_ptr()];
        // Shared across threads as addresses: raw pointers are not Send.
        let args_addr = args.as_ptr() as usize;
        let check_addr = check.as_ptr() as usize;
        let check_len = check.len();

        // The result at `cycle` is the polynomial divided by (3x)^(2^po2) - 1 for x = w^cycle, w of
        // order 2^(po2 + EXP_PO2). w^(2^po2) has order 4, so that depends only on cycle mod 4.
        let inv_zerofier: [Val; 4] = std::array::from_fn(|k| {
            let x = Val::ROU_FWD[po2 + EXP_PO2].pow(k);
            // TODO: what is this magic number 3?
            ((Val::new(3) * x).pow(1 << po2) - Val::new(1)).inv()
        });
        let write = |cycle: usize, tot: ExtVal| {
            let ret = tot * inv_zerofier[cycle % 4];
            // SAFETY: This conversion is to make the check slice mutable, which should be
            // safe because each thread access will not overlap with each other.
            let check =
                unsafe { std::slice::from_raw_parts_mut(check_addr as *mut Val, check_len) };
            for i in 0..ExtVal::EXT_SIZE {
                check[i * domain + cycle] = ret.elems()[i];
            }
        };

        match poly_fp_x16() {
            Some(poly_fp_x16) if domain % POLY_FP_LANES == 0 => {
                (0..domain / POLY_FP_LANES)
                    .into_par_iter()
                    .for_each(|block| {
                        let cycle = block * POLY_FP_LANES;
                        let mut tots = [ExtVal::ZERO; POLY_FP_LANES];
                        ffi_wrap(|| unsafe {
                            poly_fp_x16(
                                cycle,
                                domain,
                                poly_mix_pows.as_ptr(),
                                args_addr as *const *const Val,
                                tots.as_mut_ptr(),
                            )
                        })
                        .expect("eval_check");
                        for (lane, tot) in tots.into_iter().enumerate() {
                            write(cycle + lane, tot);
                        }
                    });
            }
            _ => {
                (0..domain).into_par_iter().for_each(|cycle| {
                    let mut tot = ExtVal::ZERO;
                    unsafe {
                        risc0_circuit_rv32im_cpu_poly_fp(
                            cycle,
                            domain,
                            poly_mix_pows.as_ptr(),
                            args_addr as *const *const Val,
                            &mut tot,
                        )
                    };
                    write(cycle, tot);
                });
            }
        }
    }

    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        _ctrl: &CpuBuffer<Val>,
        _global: &CpuBuffer<Val>,
        _data: &CpuBuffer<Val>,
        _mix: &CpuBuffer<Val>,
        _accum: &CpuBuffer<Val>,
        _steps: usize,
    ) {
        unimplemented!()
    }
}

#[allow(dead_code)]
pub fn segment_prover() -> Result<Box<dyn SegmentProver>> {
    let hal_factory = || {
        let suite = Poseidon2HashSuite::new_suite();
        (Arc::new(CpuHal::new(suite)), Arc::new(CpuCircuitHal))
    };
    Ok(Box::new(SegmentProverImpl::new(hal_factory)))
}
