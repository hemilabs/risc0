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

//! Metal HAL for the recursion circuit.
//!
//! Stub (docs/metal-port/PLAN.md, item F2): the factory returns the CPU prover until the
//! recursion Metal items land. The kernels are in `risc0_circuit_recursion_sys::METAL_LIB`.

use anyhow::Result;

use crate::prove::RecursionProver;

pub(crate) fn recursion_prover(hashfn: &str) -> Result<Box<dyn RecursionProver>> {
    super::cpu::recursion_prover(hashfn)
}
