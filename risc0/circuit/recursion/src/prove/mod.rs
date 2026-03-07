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

//! Prover implementation for the recursion VM.
//!
//! This module contains the recursion [Prover].

mod hal;
mod preflight;
mod program;
mod witgen;
pub mod zkr;

use std::{cell::RefCell, collections::HashMap, collections::VecDeque, fmt::Debug, rc::Rc, sync::{Arc, LazyLock}};

use anyhow::Result;
use cfg_if::cfg_if;
use risc0_core::scope;

static VERBOSE: LazyLock<bool> = LazyLock::new(|| std::env::var("RISC0_VERBOSE").is_ok());
use risc0_zkp::{
    adapter::{CircuitInfo, PROOF_SYSTEM_INFO},
    core::digest::Digest,
    field::{
        baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
        Elem as _,
    },
    hal::{Buffer, CircuitHal, Hal},
    prove::poly_group::PolyGroup,
};
use serde::{Deserialize, Serialize};

use self::{
    hal::{CircuitAccumulator, CircuitWitnessGenerator},
    witgen::WitnessGenerator,
};
use crate::{
    taps::TAPSET, CircuitImpl, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CTRL, REGISTER_GROUP_DATA,
};

pub use self::preflight::Preflight;
pub use self::program::Program;

// TODO: Automatically generate this constant from the circuit somehow without
// messing up bootstrap dependencies.
/// Size of the code group in the taps of the recursion circuit.
const RECURSION_CODE_SIZE: usize = 23;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RecursionReceipt {
    pub seal: Vec<u32>,
    pub output: Vec<u32>,
}

impl RecursionReceipt {
    /// Total number of bytes used by the seal of this receipt.
    pub fn seal_size(&self) -> usize {
        core::mem::size_of_val(self.seal.as_slice())
    }

    /// Allocates a [VecDeque] and copies the output stream into it for decoding.
    pub fn out_stream(&self) -> VecDeque<u32> {
        let mut vec: VecDeque<u32> = VecDeque::new();
        vec.extend(self.output.iter());
        vec
    }
}

pub trait RecursionProver {
    fn prove(
        &self,
        program: &Program,
        input: VecDeque<u32>,
        preflight: Option<Preflight>,
    ) -> Result<RecursionReceipt>;
}

pub fn recursion_prover(hashfn: &str) -> Result<Box<dyn RecursionProver>> {
    cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::hal::cuda::recursion_prover(hashfn)
        } else if #[cfg(feature = "rocm")] {
            self::hal::hip::recursion_prover(hashfn)
        // } else if #[cfg(any(all(target_os = "macos", target_arch = "aarch64"), target_os = "ios"))] {
        // self::hal::metal::recursion_prover(hashfn)
        } else {
            self::hal::cpu::recursion_prover(hashfn)
        }
    }
}

/// Prover for the recursion circuit.
pub struct Prover {
    program: Arc<Program>,
    hashfn: String,
    input: VecDeque<u32>,
    /// Pre-computed preflight result (set by `prepare()`).
    prepared_preflight: Option<Preflight>,
}

/// Kinds of digests recognized by the recursion program language.
// NOTE: Default is additionally a recognized type in the recursion program language. It's not
// yet supported here because some of the code in this module assumes Poseidon2 is Default.
#[non_exhaustive]
pub enum DigestKind {
    Poseidon2,
    Sha256,
}

impl Prover {
    /// Creates a new prover with the given recursion program.
    pub fn new(program: Arc<Program>, hashfn: &str) -> Self {
        Self {
            program,
            hashfn: hashfn.to_string(),
            input: VecDeque::new(),
            prepared_preflight: None,
        }
    }

    /// Pre-compute the CPU preflight. Can be called from any thread.
    /// If called before `run()`, `run()` uses the pre-computed result
    /// instead of computing the preflight on the GPU thread.
    pub fn prepare(&mut self) -> Result<()> {
        let input = std::mem::take(&mut self.input);
        let mut preflight = Preflight::new(input);
        for (cycle, row) in self.program.code_by_row().enumerate() {
            preflight.step(cycle, row)?;
        }
        self.prepared_preflight = Some(preflight);
        Ok(())
    }

    /// Add a set of u32s to the input for the recursion program.
    pub fn add_input(&mut self, input: &[u32]) {
        self.input.extend(input);
    }

    /// Add a digest to the input for the recursion program.
    pub fn add_input_digest(&mut self, digest: &Digest, kind: DigestKind) {
        match kind {
            // Poseidon2 digests consist of  BabyBear field elems and do not need to be split.
            DigestKind::Poseidon2 => self.add_input(digest.as_words()),
            // SHA-256 digests need to be split into 16-bit half words to avoid overflowing.
            DigestKind::Sha256 => self.add_input(bytemuck::cast_slice(
                &digest
                    .as_words()
                    .iter()
                    .copied()
                    .flat_map(|x| [x & 0xffff, x >> 16])
                    .map(BabyBearElem::new)
                    .collect::<Vec<_>>(),
            )),
        }
    }

    /// Run the prover, producing a receipt of execution for the recursion circuit over the loaded
    /// program and input.
    pub fn run(&mut self) -> Result<RecursionReceipt> {
        // Cache provers by hashfn to avoid recreating HAL + circuit_hal each call.
        // Using HashMap instead of single-entry cache prevents eviction when hashfn
        // alternates (e.g., "poseidon2" for lift/join → "poseidon_254" for identity_p254),
        // preserving each prover's cached ctrl PolyGroup across calls.
        thread_local! {
            static PROVER_CACHE: RefCell<HashMap<String, Box<dyn RecursionProver>>> =
                RefCell::new(HashMap::new());
        }
        let hashfn = self.hashfn.clone();
        // Take input instead of cloning to avoid copying ~4-8MB of seal data per call.
        let input = std::mem::take(&mut self.input);
        let preflight = self.prepared_preflight.take();
        PROVER_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if !cache.contains_key(&hashfn) {
                let p = recursion_prover(&hashfn)?;
                cache.insert(hashfn.clone(), p);
            }
            // Pass &Program to avoid cloning ~24MB code vector per call.
            // Pass pre-computed preflight if available (from prepare()).
            cache
                .get(&hashfn)
                .unwrap()
                .prove(&self.program, input, preflight)
        })
    }
}

pub(crate) struct RecursionProverImpl<H, C>
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
    C: CircuitHal<H> + CircuitWitnessGenerator<H>,
{
    hal: Rc<H>,
    circuit_hal: Rc<C>,
    // Cache ctrl PolyGroup by (code_rows, po2) to skip iNTT/expand/merkle on
    // repeated proofs with the same program (e.g. 43 lifts all use the same ZKR).
    cached_ctrl_group: RefCell<HashMap<(usize, usize), PolyGroup<H>>>,
    // Cache ctrl GPU buffer by (code_rows, po2) to skip re-uploading ~24MB per proof.
    cached_ctrl_buffer: RefCell<HashMap<(usize, usize), H::Buffer<H::Elem>>>,
}

impl<H, C> RecursionProver for RecursionProverImpl<H, C>
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem> + 'static,
    C: CircuitHal<H> + CircuitWitnessGenerator<H> + CircuitAccumulator<H>,
{
    fn prove(
        &self,
        program: &Program,
        input: VecDeque<u32>,
        preflight: Option<Preflight>,
    ) -> Result<RecursionReceipt> {
        scope!("prove");
        let t_total = std::time::Instant::now();

        let t0 = std::time::Instant::now();
        let (preflight, preflight_was_cached) = if let Some(pf) = preflight {
            (pf, true)
        } else {
            (self.preflight(program, input)?, false)
        };
        let preflight_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = std::time::Instant::now();
        let ctrl_key = (program.code_rows(), program.po2);
        let cached_ctrl = self.cached_ctrl_buffer.borrow().get(&ctrl_key).cloned();
        let witgen = WitnessGenerator::new(
            self.hal.as_ref(),
            self.circuit_hal.as_ref(),
            program,
            &preflight,
            cached_ctrl,
        )?;
        // Cache the ctrl GPU buffer for reuse by subsequent same-program proofs.
        if !self.cached_ctrl_buffer.borrow().contains_key(&ctrl_key) {
            self.cached_ctrl_buffer.borrow_mut().insert(ctrl_key, witgen.ctrl.clone());
        }
        let witgen_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let global = &witgen.global;

        let seal = scope!("prove", {
            let mut prover = risc0_zkp::prove::Prover::new(self.hal.as_ref(), TAPSET);
            let hashfn = &self.hal.get_hash_suite().hashfn;

            let mix = scope!("main", {
                // At the start of the protocol, seed the Fiat-Shamir transcript with context information
                // about the proof system and circuit.
                prover
                    .iop()
                    .commit(&hashfn.hash_elem_slice(&PROOF_SYSTEM_INFO.encode()));
                prover
                    .iop()
                    .commit(&hashfn.hash_elem_slice(&CircuitImpl::CIRCUIT_INFO.encode()));

                // Concat globals and po2 into a vector.
                let global_len = global.size();
                let mut header = vec![BabyBearElem::ZERO; global_len + 1];
                global.view_mut(|view| {
                    for (i, elem) in view.iter_mut().enumerate() {
                        *elem = elem.valid_or_zero();
                        header[i] = *elem;
                    }
                    header[global_len] = BabyBearElem::new_raw(program.po2 as u32);
                });

                let header_digest = hashfn.hash_elem_slice(&header);
                prover.iop().commit(&header_digest);
                prover.iop().write_field_elem_slice(header.as_slice());
                prover.set_po2(program.po2);

                // Cache ctrl PolyGroup to skip iNTT/expand/merkle on repeated proofs
                // with the same program. Lift and join ZKRs have different code_rows,
                // so we key by (code_rows, po2) to cache both independently.
                let t0 = std::time::Instant::now();
                let ctrl_key = (program.code_rows(), program.po2);
                let ctrl_cached;
                {
                    let mut cache = self.cached_ctrl_group.borrow_mut();
                    if let Some(cached) = cache.get(&ctrl_key) {
                        prover.commit_cached_group(REGISTER_GROUP_CTRL, cached.clone());
                        ctrl_cached = true;
                    } else {
                        prover.commit_group(REGISTER_GROUP_CTRL, &witgen.ctrl);
                        if let Some(group) = prover.get_group(REGISTER_GROUP_CTRL).cloned() {
                            cache.insert(ctrl_key, group);
                        }
                        ctrl_cached = false;
                    }
                }
                let ctrl_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = std::time::Instant::now();
                prover.commit_group(REGISTER_GROUP_DATA, &witgen.data);
                let data_commit_ms = t0.elapsed().as_secs_f64() * 1000.0;

                // Make the mixing values
                let mix: [BabyBearElem; CircuitImpl::MIX_SIZE] =
                    std::array::from_fn(|_| prover.iop().random_elem());

                let t0 = std::time::Instant::now();
                let mix = witgen.accum(&self.hal, self.circuit_hal.as_ref(), &mix)?;
                let accum_ms = t0.elapsed().as_secs_f64() * 1000.0;

                let t0 = std::time::Instant::now();
                prover.commit_group(REGISTER_GROUP_ACCUM, &witgen.accum);
                let accum_commit_ms = t0.elapsed().as_secs_f64() * 1000.0;

                if *VERBOSE { eprintln!("[recursion_prove] preflight={preflight_ms:.1}ms(cached={preflight_was_cached}) witgen={witgen_ms:.1}ms ctrl={ctrl_ms:.1}ms(cached={ctrl_cached}) data_commit={data_commit_ms:.1}ms accum={accum_ms:.1}ms accum_commit={accum_commit_ms:.1}ms"); }

                mix
            });

            let t0 = std::time::Instant::now();
            let seal = prover.finalize(&[&mix, global], self.circuit_hal.as_ref());
            let fri_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if *VERBOSE { eprintln!("[recursion_prove] fri={fri_ms:.1}ms total={:.1}ms", t_total.elapsed().as_secs_f64() * 1000.0); }
            seal
        });

        Ok(RecursionReceipt {
            seal,
            output: preflight.output,
        })
    }
}

impl<H, C> RecursionProverImpl<H, C>
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
    C: CircuitHal<H> + CircuitWitnessGenerator<H>,
{
    pub fn new(hal: Rc<H>, circuit_hal: Rc<C>) -> Self {
        Self {
            hal,
            circuit_hal,
            cached_ctrl_group: RefCell::new(HashMap::new()),
            cached_ctrl_buffer: RefCell::new(HashMap::new()),
        }
    }

    fn preflight(&self, program: &Program, input: VecDeque<u32>) -> Result<Preflight> {
        scope!("preflight");

        let mut preflight = Preflight::new(input);
        for (cycle, row) in program.code_by_row().enumerate() {
            preflight.step(cycle, row)?
        }

        Ok(preflight)
    }
}
