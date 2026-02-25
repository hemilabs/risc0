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

pub(crate) mod cpu;
#[cfg(feature = "cuda")]
pub(crate) mod cuda;

use std::{cell::RefCell, rc::Rc};

use anyhow::Result;
use risc0_core::scope;
use risc0_zkp::{
    adapter::{CircuitInfo as _, PROOF_SYSTEM_INFO},
    field::Elem as _,
    hal::{Buffer, CircuitHal, Hal},
    prove::{poly_group::PolyGroup, Prover},
};

use super::{
    witgen::{preflight::PreflightTrace, PreflightResults, WitnessGenerator},
    Seal, SegmentProver,
};
use crate::{
    execute::segment::Segment,
    zirgen::{
        circuit::{
            CircuitField, ExtVal, Val, REGCOUNT_MIX, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE,
            REGISTER_GROUP_DATA,
        },
        taps::TAPSET,
        CircuitImpl,
    },
    RV32IM_SEAL_VERSION,
};

pub(crate) struct MetaBuffer<H: Hal> {
    pub buf: H::Buffer<H::Elem>,
    pub rows: usize,
    pub cols: usize,
    pub checked: bool,
}

impl<H> MetaBuffer<H>
where
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal>,
{
    pub fn new(name: &'static str, hal: &H, rows: usize, cols: usize, checked: bool) -> Self {
        let buf = hal.alloc_elem_init(name, rows * cols, Val::INVALID);
        Self {
            buf,
            rows,
            cols,
            checked,
        }
    }

    #[cfg(test)]
    pub fn to_vec(&self) -> Vec<Val> {
        self.buf.to_vec()
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum StepMode {
    Parallel,
    SeqForward,
    SeqReverse,
}

pub(crate) trait CircuitWitnessGenerator<H: Hal> {
    fn generate_witness(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        global: &MetaBuffer<H>,
        data: &MetaBuffer<H>,
    ) -> Result<()>;
}

pub(crate) trait CircuitAccumulator<H: Hal> {
    fn step_accum(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<H>,
        accum: &MetaBuffer<H>,
        global: &MetaBuffer<H>,
        mix: &MetaBuffer<H>,
    ) -> Result<()>;
}

pub(crate) struct SegmentProverImpl<H, C, F>
where
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal>,
    C: CircuitHal<H> + CircuitWitnessGenerator<H>,
    F: Fn() -> (Rc<H>, Rc<C>),
{
    hal_factory: F,
    cached_hal: RefCell<Option<(Rc<H>, Rc<C>)>>,
    /// Cached code PolyGroup (always zeros, same for every segment at same po2).
    cached_code_group: RefCell<Option<(PolyGroup<H>, usize)>>,
}

impl<H, C, F> SegmentProverImpl<H, C, F>
where
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal>,
    C: CircuitHal<H> + CircuitWitnessGenerator<H>,
    F: Fn() -> (Rc<H>, Rc<C>),
{
    pub fn new(hal_factory: F) -> Self {
        Self {
            hal_factory,
            cached_hal: RefCell::new(None),
            cached_code_group: RefCell::new(None),
        }
    }

    fn get_hal(&self) -> (Rc<H>, Rc<C>) {
        let mut cached = self.cached_hal.borrow_mut();
        if let Some(ref hal) = *cached {
            return (hal.0.clone(), hal.1.clone());
        }
        let hal = (self.hal_factory)();
        *cached = Some((hal.0.clone(), hal.1.clone()));
        (hal.0, hal.1)
    }
}

impl<H, C, F> SegmentProver for SegmentProverImpl<H, C, F>
where
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal>,
    C: CircuitHal<H> + CircuitWitnessGenerator<H> + CircuitAccumulator<H>,
    F: Fn() -> (Rc<H>, Rc<C>),
{
    fn preflight(&self, segment: &Segment) -> Result<PreflightResults> {
        scope!("preflight");

        cfg_if::cfg_if! {
            if #[cfg(feature = "witgen_debug")] {
                let rand_z = ExtVal::ONE;
            } else {
                let mut rng = rand::rng();
                let rand_z = ExtVal::random(&mut rng);
            }
        }
        PreflightResults::new(segment, rand_z)
    }

    fn prove_core(&self, preflight_results: PreflightResults) -> Result<Seal> {
        scope!("prove_core");
        let t0 = std::time::Instant::now();

        cfg_if::cfg_if! {
            if #[cfg(feature = "witgen_debug")] {
                let mode = if std::env::var_os("RISC0_WITGEN_DEBUG").is_some() {
                    StepMode::SeqForward
                } else {
                    StepMode::Parallel
                };
            } else {
                let mode = StepMode::Parallel;
            }
        }

        let (hal, circuit_hal) = self.get_hal();
        eprintln!("[prove_core] hal_factory: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

        let t1 = std::time::Instant::now();
        let po2 = preflight_results.po2();
        let header = preflight_results.build_header();
        let mut witgen =
            WitnessGenerator::new(hal.as_ref(), circuit_hal.as_ref(), preflight_results, mode)?;
        eprintln!("[prove_core] witgen: {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);

        let code = &witgen.code.buf;
        let data = &witgen.data.buf;

        let t2 = std::time::Instant::now();
        let seal = scope!("prove_inner", {
            tracing::debug!("prove_inner");

            let mut prover = Prover::new(hal.as_ref(), TAPSET);
            let hashfn = &hal.get_hash_suite().hashfn;

            // Add a version tag to the start of the seal. It's not intended for
            // this value to be consumed by downstream lift predicates. Instead
            // it's meant to be a header which can be used by future versions in
            // case we'd like to adjust the encoding of the seal.
            //
            // We write via the IOP to avoid having to prepend a value to an
            // already allocated vector.
            prover.iop().write_u32_slice(&[RV32IM_SEAL_VERSION]);

            let mt0 = std::time::Instant::now();
            let (mix, global_clone) = scope!("main", {
                // At the start of the protocol, seed the Fiat-Shamir transcript with context information
                // about the proof system and circuit.
                prover
                    .iop()
                    .commit(&hashfn.hash_elem_slice(&PROOF_SYSTEM_INFO.encode()));
                prover
                    .iop()
                    .commit(&hashfn.hash_elem_slice(&CircuitImpl::CIRCUIT_INFO.encode()));

                // Use pre-computed header from CPU-side preflight data (no GPU sync needed).
                let header_digest = hashfn.hash_elem_slice(&header);
                prover.iop().commit(&header_digest);
                prover.iop().write_field_elem_slice(header.as_slice());
                prover.set_po2(po2 as usize);
                eprintln!("  [main] setup: {:.1}ms", mt0.elapsed().as_secs_f64() * 1000.0);

                // Code buffer is always zeros (INVALID → zeroized, never written to).
                // Cache its PolyGroup to skip iNTT/expand/merkle on subsequent segments
                // with the same po2. Different po2 = different domain size = must recompute.
                {
                    let mut cache = self.cached_code_group.borrow_mut();
                    if let Some((ref cached, cached_po2)) = *cache {
                        if cached_po2 == po2 as usize {
                            prover.commit_cached_group(REGISTER_GROUP_CODE, cached.clone());
                            eprintln!("  [main] commit(code) [cached]: {:.1}ms", mt0.elapsed().as_secs_f64() * 1000.0);
                        } else {
                            prover.commit_group(REGISTER_GROUP_CODE, code);
                            *cache = prover.get_group(REGISTER_GROUP_CODE).cloned()
                                .map(|g| (g, po2 as usize));
                            eprintln!("  [main] commit(code) [recomputed, po2 changed {cached_po2}->{po2}]: {:.1}ms", mt0.elapsed().as_secs_f64() * 1000.0);
                        }
                    } else {
                        prover.commit_group(REGISTER_GROUP_CODE, code);
                        *cache = prover.get_group(REGISTER_GROUP_CODE).cloned()
                            .map(|g| (g, po2 as usize));
                        eprintln!("  [main] commit(code) [computed+cached]: {:.1}ms", mt0.elapsed().as_secs_f64() * 1000.0);
                    }
                }
                prover.commit_group(REGISTER_GROUP_DATA, data);
                eprintln!("  [main] commit(data): {:.1}ms", mt0.elapsed().as_secs_f64() * 1000.0);

                // Make the mixing values
                let mix: [Val; REGCOUNT_MIX] = std::array::from_fn(|_| prover.iop().random_elem());

                let mix = witgen.accum(hal.as_ref(), circuit_hal.as_ref(), &mix)?;
                eprintln!("  [main] accum: {:.1}ms", mt0.elapsed().as_secs_f64() * 1000.0);

                // Free dead witness buffers to reduce peak GPU memory.
                // data.buf and code.buf are no longer needed after accum().
                // At po2=22 this frees ~3.3GB, enabling the accum PolyGroup allocation.
                witgen.data.buf = hal.alloc_elem("data_freed", 1);
                witgen.code.buf = hal.alloc_elem("code_freed", 1);

                prover.commit_group(REGISTER_GROUP_ACCUM, &witgen.accum.buf);
                eprintln!("  [main] commit(accum): {:.1}ms", mt0.elapsed().as_secs_f64() * 1000.0);

                // Free accum witness buffer — PolyGroup now owns the data.
                witgen.accum.buf = hal.alloc_elem("accum_freed", 1);

                // Clone tiny global buffer (90 elements = 360 bytes) so witgen can
                // be dropped during async GPU eval_check.
                let global_clone = hal.alloc_elem("global_clone", witgen.global.buf.size());
                hal.eltwise_copy_elem(&global_clone, &witgen.global.buf);

                (mix, global_clone)
            });

            // All borrows on witgen (through code, data, global, accum) are now
            // released by NLL. Move witgen into a closure that runs during GPU
            // eval_check to overlap CPU drop with GPU compute.
            let t3 = std::time::Instant::now();
            let result = prover.finalize_with_hook(
                &[&mix.buf, &global_clone],
                circuit_hal.as_ref(),
                move || {
                    let td = std::time::Instant::now();
                    drop(witgen);
                    eprintln!("[prove_core] witgen_drop (overlapped with eval_check): {:.1}ms",
                        td.elapsed().as_secs_f64() * 1000.0);
                },
            );

            eprintln!("[prove_core] prove_inner: {:.1}ms (main: {:.1}ms, finalize: {:.1}ms)",
                t2.elapsed().as_secs_f64() * 1000.0,
                (t3 - t2).as_secs_f64() * 1000.0,
                t3.elapsed().as_secs_f64() * 1000.0);
            eprintln!("[prove_core] TOTAL: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
            result
        });

        Ok(seal)
    }
}
