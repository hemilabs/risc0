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

//! The Metal eval_check translator, a `#[path]` module of build.rs.
//!
//! Stub created by item K1 (docs/metal-port/PLAN.md 4.4, 8.2); item K3 owns this file and fills
//! it in (`write_metal_eval_check`: the staged `eval_stage_<n>` kernels from the CUDA
//! eval_check sources), without editing build.rs.

use std::path::{Path, PathBuf};

/// The generated eval_check `.metal` sources to compile into the keccak metallib: written under
/// `gen_dir` (`$OUT_DIR/metal`), with `kernel_dir` (`kernels/metal`, on the include path) holding
/// hand-written headers. Called by `build_metal_kernels` only when kernels are built (not under
/// `RISC0_SKIP_BUILD_KERNELS`). Each kernel entry point it emits must also be listed in
/// `METAL_KERNEL_NAMES` (src/lib.rs). None until K3.
pub(crate) fn metal_eval_sources(_kernel_dir: &Path, _gen_dir: &Path) -> Vec<PathBuf> {
    Vec::new()
}
