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

//! Metallib loading and the compute-pipeline-state (PSO) cache (plan 2.7).

use std::collections::HashMap;

use anyhow::{anyhow, Context, Result};
use metal::{ComputePipelineState, ComputePipelineStateRef, DeviceRef, Library};

/// Z16: `RISC0_METAL_ARCHIVE_ZKP=0` builds the zkp PSOs one at a time without the binary archive
/// (the pre-Z16 loader, [MetalKernels::new]).
pub const ZKP_ARCHIVE_ENV: &str = "RISC0_METAL_ARCHIVE_ZKP";

/// Z16: binary-archive unit of the embedded zkp metallib (no other unit's prefix followed by `-`).
pub const ZKP_ARCHIVE_UNIT: &str = "zkp";

/// A loaded metallib with a PSO built for every requested kernel name.
pub struct MetalKernels {
    pub library: Library,
    pipelines: HashMap<String, ComputePipelineState>,
}

impl std::fmt::Debug for MetalKernels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut names: Vec<_> = self.pipelines.keys().collect();
        names.sort();
        f.debug_struct("MetalKernels")
            .field("pipelines", &names)
            .finish()
    }
}

impl MetalKernels {
    /// Load `bytes` as a metallib and build a PSO for each of `names`. Fails on a name the
    /// library does not contain, or on a PSO the device rejects (e.g. static threadgroup memory
    /// above 32 KB, see the D0 probes).
    pub fn new(device: &DeviceRef, bytes: &[u8], names: &[&str]) -> Result<Self> {
        let library = device
            .new_library_with_data(bytes)
            .map_err(|e| anyhow!("loading metallib ({} bytes): {e}", bytes.len()))?;
        let available = library.function_names();
        let missing: Vec<&str> = names
            .iter()
            .copied()
            .filter(|name| !available.iter().any(|a| a == name))
            .collect();
        if !missing.is_empty() {
            anyhow::bail!("metallib is missing kernel(s): {}", missing.join(", "));
        }
        let mut pipelines = HashMap::with_capacity(names.len());
        for &name in names {
            let function = library
                .get_function(name, None)
                .map_err(|e| anyhow!("{e}"))
                .with_context(|| format!("metallib function {name}"))?;
            let pso = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| anyhow!("{e}"))
                .with_context(|| format!("compute pipeline for {name}"))?;
            super::runtime::register_pso_name(&pso, name);
            pipelines.insert(name.to_string(), pso);
        }
        Ok(Self { library, pipelines })
    }

    /// Z16: [Self::new] for the process-wide zkp kernel set: the PSOs are built concurrently
    /// through the shared binary archive (`archive.rs`, unit [ZKP_ARCHIVE_UNIT]), so a process
    /// whose OS shader cache misses loads the stored GPU binaries (~10 s of sequential compiles
    /// on an M5 Max otherwise). [ZKP_ARCHIVE_ENV]`=0` restores the plain sequential build;
    /// `RISC0_METAL_ARCHIVE=0` keeps the concurrent build without an archive. Archive errors
    /// fall back to plain PSOs (never fatal), as for the circuit units.
    pub fn new_zkp(device: &DeviceRef, bytes: &[u8], names: &[&str]) -> Result<Self> {
        let archived = !std::env::var(ZKP_ARCHIVE_ENV).is_ok_and(|v| v.trim() == "0");
        Self::new_zkp_in(
            device,
            bytes,
            names,
            archived.then(super::archive::archive_dir),
        )
        .map(|(k, _)| k)
    }

    /// [Self::new_zkp] with the loader chosen by the caller (tests never set process-wide
    /// environment, PLAN 8.7): `archive` None is the plain sequential build, `Some(dir)` the
    /// archived concurrent one with the archive in `dir` (None: concurrent, no archive). Also
    /// returns the archive report (None for the plain build).
    #[doc(hidden)]
    pub fn new_zkp_in(
        device: &DeviceRef,
        bytes: &[u8],
        names: &[&str],
        archive: Option<Option<std::path::PathBuf>>,
    ) -> Result<(Self, Option<super::archive::ArchiveReport>)> {
        let Some(dir) = archive else {
            let t = std::time::Instant::now();
            let k = Self::new(device, bytes, names)?;
            tracing::debug!(
                "zkp Metal: {} PSOs in {:.2} s (sequential, no archive: {ZKP_ARCHIVE_ENV}=0)",
                names.len(),
                t.elapsed().as_secs_f64()
            );
            return Ok((k, None));
        };
        let archived =
            super::archive::ArchivedKernels::new_in(dir, device, ZKP_ARCHIVE_UNIT, bytes, names)?;
        let (library, pipelines, r) = archived.into_parts();
        tracing::info!(
            "zkp Metal: {} PSOs in {:.2} s (binary archive {}: {} hits, {} misses{})",
            names.len(),
            r.elapsed.as_secs_f64(),
            r.path
                .as_deref()
                .map_or("off".into(), |p| p.display().to_string()),
            r.hits,
            r.misses,
            if r.errors.is_empty() {
                String::new()
            } else {
                format!(", {} errors, first: {}", r.errors.len(), r.errors[0])
            }
        );
        Ok((Self { library, pipelines }, Some(r)))
    }

    /// The PSO for `name`; panics (a programming error, plan 2.9) if it was not requested at load.
    pub fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.get(name)
            .unwrap_or_else(|| panic!("Metal kernel {name} was not loaded"))
    }

    pub fn get(&self, name: &str) -> Option<&ComputePipelineState> {
        self.pipelines.get(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.pipelines.keys().map(String::as_str)
    }
}

/// SIMD-group width the cross-lane code of the zkp kernels is written for: the NTT butterflies
/// (`simd_shuffle_xor` up to lane mask 16, 8-lane transposes), the mdiv scans and the eu2
/// reductions (`simd_shuffle_down` over 32 lanes, `t & 31` / `t >> 5` lane/warp split), exactly as
/// CUDA's 32-wide warps. Every Apple GPU has 32-wide SIMD groups; a PSO reporting another
/// `threadExecutionWidth` would compute wrong results silently, so this is checked (Z8a).
pub(crate) const SIMD_WIDTH: u64 = 32;

/// Panics if `pso` (kernel `name`) does not run in [SIMD_WIDTH]-wide SIMD groups.
pub(crate) fn assert_simd_width(pso: &ComputePipelineStateRef, name: &str) {
    let width = pso.thread_execution_width();
    assert_eq!(
        width, SIMD_WIDTH,
        "Metal kernel {name}: threadExecutionWidth {width}, the kernel assumes {SIMD_WIDTH}"
    );
}
