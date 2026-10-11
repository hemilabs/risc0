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

use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use risc0_zkp::field::Elem as _;

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
    _hal: Arc<IntelHal<IH>>, // retain a reference to ensure the context remains valid
    // Host shadows (RISC0_RECURSION_HOST_SHADOW, default on). The CPU witgen and
    // accum steps run on host copies of device buffers; the device copies were
    // either produced on the host in the first place (data/accum after witgen)
    // or never change (ctrl), so the ~150 MB of D2H per step is avoidable.
    //  * ctrl: read-only for witgen/accum (no generated step writes args[0]);
    //    cached per device buffer. The cache holds a clone of the buffer so its
    //    device pointer cannot be recycled by the pool while the entry lives.
    //  * data: after witgen the host vector is exactly the uploaded device
    //    content; generic code then writes ZK noise into the tail ZK_CYCLES
    //    rows of every column and zeroizes INVALID. accumulate() re-reads just
    //    those tail rows from the device and applies the same zeroize on host.
    ctrl_cache: Mutex<Vec<(usize, usize, IntelBuffer<BabyBearElem>, Arc<Vec<BabyBearElem>>)>>,
    data_shadow: Mutex<Option<(usize, usize, Vec<BabyBearElem>)>>,
}

impl<IH: IntelHash> IntelCircuitHal<IH> {
    pub fn new(_hal: Arc<IntelHal<IH>>) -> Self {
        Self {
            _hal,
            ctrl_cache: Mutex::new(Vec::new()),
            data_shadow: Mutex::new(None),
        }
    }
}

fn host_shadow_enabled() -> bool {
    !matches!(std::env::var("RISC0_RECURSION_HOST_SHADOW").as_deref(), Ok("0"))
}

fn shadow_check_enabled() -> bool {
    std::env::var_os("RISC0_RECURSION_SHADOW_CHECK").is_some()
}

fn buf_key<T: Clone>(buf: &IntelBuffer<T>) -> (usize, usize) {
    (buf.as_device_ptr().0 as usize, buf.size())
}

/// Blocking D2H of `dst.len()` elements starting at element `offset` of `buf`.
fn dtoh_into(buf: &IntelBuffer<BabyBearElem>, offset: usize, dst: &mut [BabyBearElem]) {
    assert!(offset + dst.len() <= buf.size());
    let src = buf.as_device_ptr_with_offset(offset);
    unsafe {
        risc0_sys::intel::esimd_memcpy_dtoh(
            risc0_sys::intel::get_queue(),
            dst.as_mut_ptr() as *mut std::ffi::c_void,
            src.0 as *const std::ffi::c_void,
            std::mem::size_of_val(dst),
        );
    }
}

/// Single-copy D2H (BufferImpl::to_vec stages through a second host vector).
fn dtoh_vec(buf: &IntelBuffer<BabyBearElem>) -> Vec<BabyBearElem> {
    let mut v = vec_filled(buf.size(), BabyBearElem::ZERO);
    dtoh_into(buf, 0, &mut v);
    v
}

fn vec_filled(n: usize, val: BabyBearElem) -> Vec<BabyBearElem> {
    let mut v: Vec<BabyBearElem> = Vec::with_capacity(n);
    v.spare_capacity_mut()
        .par_chunks_mut(1 << 16)
        .for_each(|c| c.iter_mut().for_each(|x| { x.write(val); }));
    unsafe { v.set_len(n) };
    v
}

/// Refresh the tail ZK_CYCLES rows of every column from the device.
fn refresh_tail_rows(buf: &IntelBuffer<BabyBearElem>, host: &mut [BabyBearElem], total: usize) {
    use risc0_zkp::ZK_CYCLES;
    let cols = host.len() / total;
    for c in 0..cols {
        let off = c * total + total - ZK_CYCLES;
        dtoh_into(buf, off, &mut host[off..off + ZK_CYCLES]);
    }
}

fn zeroize_host(v: &mut [BabyBearElem]) {
    v.par_chunks_mut(1 << 16)
        .for_each(|c| c.iter_mut().for_each(|x| *x = x.valid_or_zero()));
}

fn shadow_compare(name: &str, buf: &IntelBuffer<BabyBearElem>, host: &[BabyBearElem]) -> Result<()> {
    let dev = dtoh_vec(buf);
    let bad = dev
        .iter()
        .zip(host.iter())
        .filter(|(a, b)| a.as_u32_montgomery() != b.as_u32_montgomery())
        .count();
    if bad != 0 || dev.len() != host.len() {
        bail!("host shadow mismatch on {name}: {bad} of {} differ", dev.len());
    }
    eprintln!("[recursion_shadow_check] {name}: {} elems identical", dev.len());
    Ok(())
}

impl<IH: IntelHash> IntelCircuitHal<IH> {
    fn host_ctrl(&self, ctrl: &IntelBuffer<BabyBearElem>) -> Arc<Vec<BabyBearElem>> {
        let (ptr, len) = buf_key(ctrl);
        let mut cache = self.ctrl_cache.lock().unwrap();
        if let Some(e) = cache.iter().find(|e| e.0 == ptr && e.1 == len) {
            return e.3.clone();
        }
        let v = Arc::new(dtoh_vec(ctrl));
        // Bounded: the entry holds a clone of the device buffer (so its pointer
        // cannot be recycled while cached); evicting drops that clone, after
        // which the pointer may be reused - but then it is no longer a key here.
        if cache.len() >= 8 {
            cache.remove(0);
        }
        cache.push((ptr, len, ctrl.clone(), v.clone()));
        v
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
        let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
        let shadow = host_shadow_enabled();
        let t0 = std::time::Instant::now();
        let (ctrl_host, mut data_host, mut global_host) = if shadow {
            // data was just allocated with alloc_elem_init(INVALID).
            let data_host = vec_filled(data.size(), BabyBearElem::INVALID);
            if shadow_check_enabled() {
                shadow_compare("data(pre-witgen)", data, &data_host)?;
            }
            (self.host_ctrl(ctrl), data_host, dtoh_vec(global))
        } else {
            (Arc::new(ctrl.to_vec()), data.to_vec(), global.to_vec())
        };
        let d2h_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let buffers = RawExecBuffers {
            ctrl: ctrl_host.as_ptr(),
            data: data_host.as_mut_ptr(),
            global: global_host.as_mut_ptr(),
        };
        let t1 = std::time::Instant::now();
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_cpu_witgen(mode, &buffers, preflight, total_cycles)
        })?;
        let ffi_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = std::time::Instant::now();
        if !shadow {
            ctrl.copy_from_host(&ctrl_host);
        }
        data.copy_from_host(&data_host);
        global.copy_from_host(&global_host);
        let h2d_ms = t2.elapsed().as_secs_f64() * 1000.0;

        if shadow {
            let (ptr, len) = buf_key(data);
            *self.data_shadow.lock().unwrap() = Some((ptr, len, data_host));
        } else {
            *self.data_shadow.lock().unwrap() = None;
        }

        if verbose {
            eprintln!("[recursion_witgen_intel] d2h={d2h_ms:.1}ms ffi_cpu={ffi_ms:.1}ms h2d={h2d_ms:.1}ms total={:.1}ms cycles={} shadow={shadow}",
                t0.elapsed().as_secs_f64() * 1000.0,
                total_cycles);
        }

        Ok(())
    }

    fn upload_host_witness(
        &self,
        data: &IntelBuffer<BabyBearElem>,
        global: &IntelBuffer<BabyBearElem>,
        host_data: Vec<BabyBearElem>,
        host_global: Vec<BabyBearElem>,
    ) -> Result<()> {
        let t0 = std::time::Instant::now();
        data.copy_from_host(&host_data);
        global.copy_from_host(&host_global);
        if host_shadow_enabled() {
            let (ptr, len) = buf_key(data);
            *self.data_shadow.lock().unwrap() = Some((ptr, len, host_data));
        } else {
            *self.data_shadow.lock().unwrap() = None;
        }
        if std::env::var_os("RISC0_VERBOSE").is_some() {
            eprintln!("[recursion_witgen_intel] prewitgen upload h2d={:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
        }
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
        let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
        let t0 = std::time::Instant::now();
        let total = total_cycles as usize;
        let shadow_data = if host_shadow_enabled() {
            let key = buf_key(data);
            match self.data_shadow.lock().unwrap().take() {
                Some((p, l, v)) if (p, l) == key => Some(v),
                _ => None,
            }
        } else {
            None
        };
        let shadow = shadow_data.is_some();
        let (ctrl_host, global_host, data_host, mix_host, mut accum_host) =
            if let Some(mut data_host) = shadow_data {
                // dtoh_into waits on the in-order queue, so every earlier H2D
                // from data_host has completed before we mutate it.
                refresh_tail_rows(data, &mut data_host, total);
                zeroize_host(&mut data_host);
                // accum: alloc_elem_init(INVALID) + ZK noise in the tail rows.
                let mut accum_host = vec_filled(accum.size(), BabyBearElem::INVALID);
                refresh_tail_rows(accum, &mut accum_host, total);
                if shadow_check_enabled() {
                    shadow_compare("data(accum)", data, &data_host)?;
                    shadow_compare("accum(pre)", accum, &accum_host)?;
                }
                (self.host_ctrl(ctrl), dtoh_vec(global), data_host, dtoh_vec(mix), accum_host)
            } else {
                (
                    Arc::new(ctrl.to_vec()),
                    global.to_vec(),
                    data.to_vec(),
                    mix.to_vec(),
                    accum.to_vec(),
                )
            };
        let d2h_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let buffers = RawAccumBuffers {
            ctrl: ctrl_host.as_ptr(),
            global: global_host.as_ptr(),
            data: data_host.as_ptr(),
            mix: mix_host.as_ptr(),
            accum: accum_host.as_mut_ptr(),
        };
        let t1 = std::time::Instant::now();
        ffi_wrap(|| unsafe {
            risc0_circuit_recursion_cpu_accum(&buffers, work_cycles, total_cycles)
        })?;
        let ffi_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // Upload modified buffers back to GPU (H2D only, no redundant D2H).
        // The accum steps write only args[4] (accum), so global is unchanged.
        let t2 = std::time::Instant::now();
        accum.copy_from_host(&accum_host);
        if !shadow {
            global.copy_from_host(&global_host);
        }
        let h2d_ms = t2.elapsed().as_secs_f64() * 1000.0;

        if verbose {
            eprintln!("[recursion_accum_intel] d2h={d2h_ms:.1}ms ffi_cpu={ffi_ms:.1}ms h2d={h2d_ms:.1}ms total={:.1}ms work_cycles={} total_cycles={} shadow={shadow}",
                t0.elapsed().as_secs_f64() * 1000.0,
                work_cycles, total_cycles);
        }

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
            let hal = Arc::new(IntelHalPoseidon2::new());
            let circuit_hal = Arc::new(IntelCircuitHalPoseidon2::new(hal.clone()));
            Ok(Box::new(RecursionProverImpl::new(hal, circuit_hal)))
        }
        "sha-256" => {
            let hal = Arc::new(IntelHalSha256::new());
            let circuit_hal = Arc::new(IntelCircuitHalSha256::new(hal.clone()));
            Ok(Box::new(RecursionProverImpl::new(hal, circuit_hal)))
        }
        "poseidon_254" => {
            // GPU Poseidon-254 via ESIMD — BN254 field arithmetic on Intel Arc.
            // hash_fold/hash_rows run entirely on GPU with zero PCIe round-trips.
            let hal = Arc::new(IntelHalPoseidon254::new());
            let circuit_hal = Arc::new(IntelCircuitHalPoseidon254::new(hal.clone()));
            Ok(Box::new(RecursionProverImpl::new(hal, circuit_hal)))
        }
        _ => bail!("Unsupported hashfn: {hashfn}"),
    }
}
