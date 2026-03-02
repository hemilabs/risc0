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

use std::{cell::RefCell, collections::VecDeque, fmt::Debug, mem, rc::Rc};

use anyhow::Result;
use cfg_if::cfg_if;
use risc0_core::scope;
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
    preflight::Preflight,
    witgen::WitnessGenerator,
};
use crate::{
    taps::TAPSET, CircuitImpl, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CTRL, REGISTER_GROUP_DATA,
};

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
    fn prove(&self, program: Program, input: VecDeque<u32>) -> Result<RecursionReceipt>;
}

pub fn recursion_prover(hashfn: &str) -> Result<Box<dyn RecursionProver>> {
    cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::hal::cuda::recursion_prover(hashfn)
        // } else if #[cfg(any(all(target_os = "macos", target_arch = "aarch64"), target_os = "ios"))] {
        // self::hal::metal::recursion_prover(hashfn)
        } else {
            self::hal::cpu::recursion_prover(hashfn)
        }
    }
}

thread_local! {
    /// Cache of pre-computed preflights, keyed by (po2, code_len).
    /// Populated by `push_preflight_handles()` and consumed by `RecursionProverImpl::prove()`.
    static PREFLIGHT_CACHE: RefCell<Vec<(Preflight, usize, usize)>> = RefCell::new(Vec::new());
}

/// Opaque handle for a pre-computed preflight result.
/// Created by `Prover::compute_preflight()` and consumed by `push_preflight_handles()`.
pub struct PreflightHandle {
    preflight: Preflight,
    po2: usize,
    code_len: usize,
}

// Safety: Preflight contains only owned data (Vecs, BTreeMaps, scalars).
unsafe impl Send for PreflightHandle {}

/// Push pre-computed preflight handles into the thread-local cache.
/// Must be called from the same thread that will later call `Prover::run()`.
pub fn push_preflight_handles(handles: Vec<PreflightHandle>) {
    PREFLIGHT_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        for h in handles {
            cache.push((h.preflight, h.po2, h.code_len));
        }
    });
}

/// Clear any remaining pre-computed preflights from the cache.
pub fn clear_preflight_cache() {
    PREFLIGHT_CACHE.with(|cache| cache.borrow_mut().clear());
}

/// Prover for the recursion circuit.
pub struct Prover {
    program: Program,
    hashfn: String,
    input: VecDeque<u32>,
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
    pub fn new(program: Program, hashfn: &str) -> Self {
        Self {
            program,
            hashfn: hashfn.to_string(),
            input: VecDeque::new(),
        }
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

    /// Pre-compute the preflight for this prover.
    /// Returns a handle that can be passed to `push_preflight_handles()`.
    /// The input is consumed; `run()` will use the cached preflight.
    pub fn compute_preflight(&mut self) -> Result<PreflightHandle> {
        scope!("compute_preflight");
        let mut preflight = Preflight::new(mem::take(&mut self.input));
        for (cycle, row) in self.program.code_by_row().enumerate() {
            preflight.step(cycle, row)?
        }
        Ok(PreflightHandle {
            po2: self.program.po2,
            code_len: self.program.code.len(),
            preflight,
        })
    }

    /// Run the prover, producing a receipt of execution for the recursion circuit over the loaded
    /// program and input.
    pub fn run(&mut self) -> Result<RecursionReceipt> {
        thread_local! {
            static CACHED: RefCell<Option<(String, Box<dyn RecursionProver>)>> =
                RefCell::new(None);
        }
        CACHED.with(|cache| {
            let mut cache = cache.borrow_mut();
            let need_new = match cache.as_ref() {
                Some((hashfn, _)) => hashfn != &self.hashfn,
                None => true,
            };
            if need_new {
                *cache = Some((self.hashfn.clone(), recursion_prover(&self.hashfn)?));
            }
            let (_, prover) = cache.as_ref().unwrap();
            prover.prove(mem::take(&mut self.program), mem::take(&mut self.input))
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
    /// Cached ctrl PolyGroups keyed by (po2, code_len). The ctrl group is deterministic
    /// for a given recursion program, so repeat calls with the same program
    /// skip iNTT/expand/merkle recomputation. Stored as a Vec since there are
    /// only a few distinct programs (lift, join, identity_p254, shrink_wrap).
    cached_ctrl_groups: RefCell<Vec<(PolyGroup<H>, usize, usize)>>,
    /// Cached transposed ctrl CPU buffers keyed by (po2, code_len).
    /// Avoids re-transposing program code into column-major layout on repeat calls.
    cached_ctrl_cpu: RefCell<Vec<(Vec<BabyBearElem>, usize, usize)>>,
    /// Cached ctrl GPU buffers keyed by (po2, code_len).
    /// Avoids re-uploading ~24MB from CPU→GPU (pageable memcpy) every proof.
    cached_ctrl_gpu: RefCell<Vec<(H::Buffer<H::Elem>, usize, usize)>>,
}

impl<H, C> RecursionProver for RecursionProverImpl<H, C>
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
    C: CircuitHal<H> + CircuitWitnessGenerator<H> + CircuitAccumulator<H>,
{
    fn prove(&self, program: Program, input: VecDeque<u32>) -> Result<RecursionReceipt> {
        scope!("prove");

        // Use pre-computed preflight from cache if available (from batch_preflight_lifts).
        let preflight = {
            let cached = PREFLIGHT_CACHE.with(|cache| {
                let mut cache = cache.borrow_mut();
                if let Some(idx) = cache.iter().position(|(_, po2, code_len)| {
                    *po2 == program.po2 && *code_len == program.code.len()
                }) {
                    Some(cache.remove(idx).0)
                } else {
                    None
                }
            });
            match cached {
                Some(pf) => pf,
                None => self.preflight(&program, input)?,
            }
        };

        // Ensure the transposed ctrl CPU buffer is cached, then borrow it.
        {
            let needs_compute = !self
                .cached_ctrl_cpu
                .borrow()
                .iter()
                .any(|(_, po2, code_len)| {
                    *po2 == program.po2 && *code_len == program.code.len()
                });
            if needs_compute {
                let total_cycles = 1usize << program.po2;
                let ctrl_size = program.code_size;
                let mut ctrl = vec![BabyBearElem::ZERO; total_cycles * ctrl_size];
                for i in 0..program.code_rows() {
                    for j in 0..ctrl_size {
                        ctrl[j * total_cycles + i] = program.code[i * ctrl_size + j];
                    }
                }
                self.cached_ctrl_cpu.borrow_mut().push((
                    ctrl,
                    program.po2,
                    program.code.len(),
                ));
            }
        }
        let ctrl_cache = self.cached_ctrl_cpu.borrow();
        let ctrl_transposed = &ctrl_cache
            .iter()
            .find(|(_, po2, code_len)| {
                *po2 == program.po2 && *code_len == program.code.len()
            })
            .unwrap()
            .0;

        // Use cached GPU ctrl buffer if available (avoids ~2ms CPU→GPU pageable memcpy).
        let cached_ctrl_gpu = {
            let cache = self.cached_ctrl_gpu.borrow();
            cache
                .iter()
                .find(|(_, po2, code_len)| {
                    *po2 == program.po2 && *code_len == program.code.len()
                })
                .map(|(buf, _, _)| buf.clone())
        };

        let witgen = WitnessGenerator::new(
            self.hal.as_ref(),
            self.circuit_hal.as_ref(),
            &program,
            &preflight,
            ctrl_transposed,
            cached_ctrl_gpu.as_ref(),
        )?;

        // Cache the GPU ctrl buffer after first upload.
        if cached_ctrl_gpu.is_none() {
            let ctrl_clone = self.hal.alloc_elem("ctrl_cache", witgen.ctrl.size());
            self.hal.eltwise_copy_elem(&ctrl_clone, &witgen.ctrl);
            self.cached_ctrl_gpu.borrow_mut().push((
                ctrl_clone,
                program.po2,
                program.code.len(),
            ));
        }

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

                // Cache the ctrl PolyGroup: ctrl is deterministic per (program, po2),
                // so repeat calls with the same program skip iNTT/expand/merkle.
                {
                    let cached_ctrl = {
                        let cache = self.cached_ctrl_groups.borrow();
                        cache
                            .iter()
                            .find(|(_, po2, code_len)| {
                                *po2 == program.po2 && *code_len == program.code.len()
                            })
                            .map(|(group, _, _)| group.clone())
                    };
                    if let Some(group) = cached_ctrl {
                        prover.commit_cached_group(REGISTER_GROUP_CTRL, group);
                    } else {
                        prover.commit_group(REGISTER_GROUP_CTRL, &witgen.ctrl);
                        if let Some(group) = prover.get_group(REGISTER_GROUP_CTRL).cloned() {
                            self.cached_ctrl_groups.borrow_mut().push((
                                group,
                                program.po2,
                                program.code.len(),
                            ));
                        }
                    }
                }
                prover.commit_group(REGISTER_GROUP_DATA, &witgen.data);

                // Make the mixing values
                let mix: [BabyBearElem; CircuitImpl::MIX_SIZE] =
                    std::array::from_fn(|_| prover.iop().random_elem());

                let mix = witgen.accum(&self.hal, self.circuit_hal.as_ref(), &mix)?;

                prover.commit_group(REGISTER_GROUP_ACCUM, &witgen.accum);

                mix
            });
            prover.finalize(&[&mix, global], self.circuit_hal.as_ref())
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
            cached_ctrl_groups: RefCell::new(Vec::new()),
            cached_ctrl_cpu: RefCell::new(Vec::new()),
            cached_ctrl_gpu: RefCell::new(Vec::new()),
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
