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

//! Where the groth16 pipeline states come from (item G11).
//!
//! Cold creation of the groth16 PSOs (the OS shader cache misses: a newly built executable, an
//! evicted or disabled cache) compiles the MSM point kernels, the slowest of every unit: it set
//! the worker's ~25 s cold startup (BASELINES.md section 3) and took minutes in G7's runs. By
//! default the PSOs of the default path ([ARCHIVED_PSOS]: small kernels, MSM, NTT) are built
//! through the zkp HAL's `MTLBinaryArchive` ([risc0_zkp::hal::metal::archive], unit
//! [ARCHIVE_UNIT]; key = metallib digest + device + OS build + names, integrity-checked, keep-N
//! cleanup, fallback to a plain build on any archive error), concurrently: a hit loads the stored
//! GPU binary without compiling. The revert-path pipelines (`_g1`/`_g2` MSM point kernels, G10)
//! stay lazy and are built plainly on first use.
//!
//! `RISC0_G16_METAL_ARCHIVE=0` restores the pre-G11 behaviour exactly: serial plain builds of the
//! small and MSM PSOs in `kernels()`, and the NTT PSOs built with each NTT plan.
//! (`RISC0_METAL_ARCHIVE=0` and `RISC0_METAL_ARCHIVE_DIR` of the shared module apply as well: with
//! no archive directory the PSOs are still built concurrently once, but nothing is stored.)

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{anyhow, Result};
use metal::{ComputePipelineState, DeviceRef, LibraryRef};
use risc0_zkp::hal::metal::archive::{ArchiveReport, ArchivedKernels};

/// `0` disables the groth16 binary archive (the pre-G11 pipeline loader).
pub const ARCHIVE_ENV: &str = "RISC0_G16_METAL_ARCHIVE";

/// The unit name of the groth16 archive (not a `-`-prefix of another unit's, nor they of it).
pub const ARCHIVE_UNIT: &str = "groth16";

/// The small kernels' PSOs ([super::small::SmallKernels]).
pub const SMALL_PSOS: &[&str] = &[
    "fr_mul",
    "fr_mul_sub_from_mont",
    "fr_add",
    "witness_into_poly",
    "witness_into_poly_compact",
    "chacha_fuzz",
    "fr_powers",
    "fr_scale",
];

/// The MSM PSOs built eagerly ([super::msm::MsmKernels]; the `_g1`/`_g2` ones are lazy).
pub const MSM_PSOS: &[&str] = &[
    "msm_clear",
    "msm_hist_count",
    "msm_window_scan",
    "msm_scatter",
    "msm_buckets_g1w",
    "msm_segment_reduce_g1w",
    "msm_window_reduce_g1w",
    "msm_buckets_g2w",
    "msm_segment_reduce_g2w",
    "msm_window_reduce_g2w",
];

/// The NTT PSOs ([super::ntt::NttKernels]; all of them, the 3 non-`_w` ones size the fused block).
pub const NTT_PSOS: &[&str] = &[
    "bit_reverse",
    "ntt_stage",
    "ntt_fused_stages",
    "ntt_powers",
    "ntt_mul_powers",
    "ntt_scale",
    "ntt_stage_w",
    "ntt_fused_stages_w",
    "ntt_mul_powers_w",
];

/// Every PSO the archive holds: [SMALL_PSOS], [MSM_PSOS], [NTT_PSOS] (part of the archive key).
pub fn archived_psos() -> Vec<&'static str> {
    [SMALL_PSOS, MSM_PSOS, NTT_PSOS].concat()
}

/// Whether `kernels()` builds through the archive ([ARCHIVE_ENV] is not `0`).
pub fn use_archive() -> bool {
    !std::env::var(ARCHIVE_ENV).is_ok_and(|v| v.trim() == "0")
}

/// A source of PSOs for the kernel sets' constructors: built plainly from a library (the pre-G11
/// way, and the lazy revert pipelines), or taken from an [ArchivedKernels].
pub enum PsoSource<'a> {
    Plain {
        device: &'a DeviceRef,
        lib: &'a LibraryRef,
    },
    Archived {
        device: &'a DeviceRef,
        kernels: &'a ArchivedKernels,
        /// PSOs requested that the archived set does not hold (built plainly; a test keeps it 0).
        outside: AtomicUsize,
    },
}

impl<'a> PsoSource<'a> {
    pub fn plain(device: &'a DeviceRef, lib: &'a LibraryRef) -> Self {
        Self::Plain { device, lib }
    }

    pub fn archived(device: &'a DeviceRef, kernels: &'a ArchivedKernels) -> Self {
        Self::Archived {
            device,
            kernels,
            outside: AtomicUsize::new(0),
        }
    }

    /// The PSO of kernel `name`.
    pub fn pso(&self, name: &str) -> Result<ComputePipelineState> {
        match self {
            Self::Plain { device, lib } => plain_pso(device, lib, name),
            Self::Archived {
                device,
                kernels,
                outside,
            } => match kernels.get(name) {
                Some(p) => Ok(p.clone()),
                None => {
                    outside.fetch_add(1, Ordering::Relaxed);
                    plain_pso(device, kernels.library(), name)
                }
            },
        }
    }

    /// How many requested PSOs were not in the archived set.
    pub fn outside(&self) -> usize {
        match self {
            Self::Plain { .. } => 0,
            Self::Archived { outside, .. } => outside.load(Ordering::Relaxed),
        }
    }
}

/// `newComputePipelineStateWithFunction` for `name` of `lib` (the pre-G11 build).
pub fn plain_pso(device: &DeviceRef, lib: &LibraryRef, name: &str) -> Result<ComputePipelineState> {
    let f = lib
        .get_function(name, None)
        .map_err(|e| anyhow!("groth16 metallib: kernel {name}: {e}"))?;
    device
        .new_compute_pipeline_state_with_function(&f)
        .map_err(|e| anyhow!("groth16 metallib: pipeline {name}: {e}"))
}

/// Build [archived_psos] from `bytes` through the archive in `dir` (None: no archive, a
/// concurrent plain build).
pub fn build_archived(
    device: &DeviceRef,
    dir: Option<std::path::PathBuf>,
    bytes: &[u8],
) -> Result<ArchivedKernels> {
    ArchivedKernels::new_in(dir, device, ARCHIVE_UNIT, bytes, &archived_psos())
}

/// A one-line summary of an archive report (`RISC0_VERBOSE` / the worker log).
pub fn describe(report: &ArchiveReport) -> String {
    format!(
        "{} PSOs in {:.2} s (archive {}: {} hits, {} misses, saved {}{})",
        report.hits + report.misses,
        report.elapsed.as_secs_f64(),
        report
            .path
            .as_deref()
            .map_or("off".into(), |p| p.display().to_string()),
        report.hits,
        report.misses,
        report.saved,
        if report.errors.is_empty() {
            String::new()
        } else {
            format!(", errors: {}", report.errors.join("; "))
        },
    )
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::metal::{msm::MsmKernels, ntt::NttKernels, small::SmallKernels};

    fn device() -> Option<metal::Device> {
        if crate::METAL_LIB.is_empty() {
            return None;
        }
        metal::Device::system_default()
    }

    /// The archived set is exactly what the default-path constructors request: none falls
    /// outside it, every listed name is a kernel of the library, and none is listed twice.
    #[test]
    fn archived_set_covers_the_default_path() {
        let _lock = crate::metal::test_lock::shared();
        let Some(device) = device() else {
            return;
        };
        let names = archived_psos();
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "a PSO is listed twice");
        for n in &names {
            assert!(crate::METAL_KERNEL_NAMES.contains(n), "{n} is not a kernel");
        }
        let k = build_archived(&device, None, crate::METAL_LIB).unwrap();
        let src = PsoSource::archived(&device, &k);
        SmallKernels::from_source(&src).unwrap();
        MsmKernels::from_source(&device, k.library(), &src).unwrap();
        NttKernels::from_source(&src).unwrap();
        assert_eq!(src.outside(), 0, "a default-path PSO is not archived");
    }

    /// Save to a fresh directory, then load from it in a second build: every PSO is a hit, and
    /// exactly one archive file is written.
    #[test]
    fn archive_roundtrip() {
        let _lock = crate::metal::test_lock::shared();
        let Some(device) = device() else {
            return;
        };
        let dir = std::env::temp_dir().join(format!("g11-archive-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let n = archived_psos().len();
        let first = build_archived(&device, Some(dir.clone()), crate::METAL_LIB).unwrap();
        assert!(
            first.report.errors.is_empty(),
            "{}",
            describe(&first.report)
        );
        assert_eq!(first.report.hits + first.report.misses, n);
        assert!(first.report.saved, "{}", describe(&first.report));
        let second = build_archived(&device, Some(dir.clone()), crate::METAL_LIB).unwrap();
        assert!(second.report.loaded, "{}", describe(&second.report));
        assert_eq!(second.report.hits, n, "{}", describe(&second.report));
        assert!(!second.report.saved);
        let files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "binarchive"))
            .collect();
        assert_eq!(files.len(), 1, "{files:?}");
        assert!(files[0]
            .file_name()
            .to_string_lossy()
            .starts_with(&format!("{ARCHIVE_UNIT}-")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Cold-start probe (G11; run with `MTL_SHADER_CACHE_SIZE=0` in a fresh process, one arm per
    /// process). `G11_ARM`: `plain` (serial plain build of every archived PSO, per set, the
    /// pre-G11 order and per-PSO times), `archive` (through the archive in `G11_DIR`, default the
    /// production directory). Prints one `G11 ...` line.
    #[test]
    #[ignore]
    fn g11_pso_cold_times() {
        let _lock = crate::metal::test_lock::exclusive();
        let device = device().expect("Metal device and kernels");
        let arm = std::env::var("G11_ARM").unwrap_or_else(|_| "plain".into());
        let t0 = Instant::now();
        match arm.as_str() {
            "plain" => {
                let lib = device.new_library_with_data(crate::METAL_LIB).unwrap();
                let mut per = Vec::new();
                for set in [SMALL_PSOS, MSM_PSOS, NTT_PSOS] {
                    let ts = Instant::now();
                    for name in set {
                        let t = Instant::now();
                        plain_pso(&device, &lib, name).unwrap();
                        per.push(format!("{name}={:.2}", t.elapsed().as_secs_f64()));
                    }
                    println!("G11 set {:.2} s", ts.elapsed().as_secs_f64());
                }
                println!(
                    "G11 arm=plain total={:.2} s per: {}",
                    t0.elapsed().as_secs_f64(),
                    per.join(" ")
                );
            }
            "archive" => {
                let dir = std::env::var_os("G11_DIR")
                    .map(std::path::PathBuf::from)
                    .or_else(risc0_zkp::hal::metal::archive::archive_dir);
                let k = build_archived(&device, dir, crate::METAL_LIB).unwrap();
                println!(
                    "G11 arm=archive total={:.2} s {}",
                    t0.elapsed().as_secs_f64(),
                    describe(&k.report)
                );
            }
            other => panic!("G11_ARM={other}"),
        }
    }

    /// The production entry point, cold (G11 A/B; `MTL_SHADER_CACHE_SIZE=0`, one process per
    /// run, the arm chosen by the process environment: `RISC0_G16_METAL_ARCHIVE=0` is the pre-G11
    /// loader): `kernels()` (what `metal_warmup` builds) plus the NTT plan of the rzup key's
    /// domain (2^23, what the first prover build adds). Prints one `G11 kernels` line.
    #[test]
    #[ignore]
    fn g11_kernels_cold() {
        let _lock = crate::metal::test_lock::exclusive();
        let t0 = Instant::now();
        let k = crate::metal::prover::kernels().unwrap();
        let t_k = t0.elapsed();
        let t1 = Instant::now();
        let plan = k.ntt_plan(23).unwrap();
        let t_ntt = t1.elapsed();
        drop(plan);
        println!(
            "G11 kernels archive={} kernels={:.3} s ntt_plan={:.3} s total={:.3} s {}",
            use_archive(),
            t_k.as_secs_f64(),
            t_ntt.as_secs_f64(),
            t0.elapsed().as_secs_f64(),
            k.archive.as_ref().map_or("off".into(), describe)
        );
    }
}
