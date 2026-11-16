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

//! GPU-binary caching of circuit pipelines with `MTLBinaryArchive` (PLAN 9.1; keccak item K5,
//! hoisted here from `risc0-circuit-keccak`'s `hal/metal/archive.rs` by R14 so that keccak,
//! rv32im and recursion share one implementation, each under its own unit name).
//!
//! Creating a circuit's PSOs compiles the metallib's AIR to GPU code: 40-190 s cold for the 58
//! keccak eval_check stages and 20-35 s for its 12 witgen arms (K3, K4, K6 notes), ~30-45 s for
//! rv32im's evalfast `ef_eval_split` plus several seconds for its witgen/accum kernels (R5, R12,
//! R15e notes) on an M5 Max, more on the 8-10-core base chips of the minimum target. The OS
//! shader cache usually makes a second process instant, but it was seen to miss (cache shared
//! with every Metal app, evictions, relinked executables), so the worker cannot rely on it to
//! keep cold PSO creation out of a job deadline. This module keeps the compiled GPU binaries in a
//! binary archive file of our own:
//!
//! - **Location**: `RISC0_METAL_ARCHIVE_DIR`, else the risc0 cache directory
//!   (`~/Library/Caches/com.risczero.RISC-Zero.risc0`, the directory `risc0-build-kernel` caches
//!   compiler outputs in) `/metal-archives`.
//! - **Key**: `<unit>-<sha256(digest(metallib) ‖ device name ‖ OS build ‖ kernel names)[..24]>`,
//!   `digest` = [chunked_digest] (K7: sha256 of the per-MiB sha256s, hashed on all cores).
//!   A new metallib, another GPU or an OS update (which can change the GPU compiler) therefore
//!   never reads an old archive. When a new one is written, the unit keeps its
//!   `RISC0_METAL_ARCHIVE_KEEP` (default 4) most recently used other keys and deletes the rest,
//!   and any key unused for 30 days ([prune_other_keys]; K7, K8 finding 4). Units must not be
//!   prefixes of each other followed by `-` (`keccak`, `rv32im`, `recursion`, `groth16` are not;
//!   `groth16` is `risc0-groth16-sys`'s, item G11).
//! - **Integrity**: the file is `<unit>-<key>-<digest(file)[..16]>.binarchive`, and its content
//!   is checked against that name before Metal sees it. Metal does not survive every damaged
//!   archive: a file with some bytes flipped loads, and then creating a PSO from it aborts the
//!   process with an uncaught C++ `std::length_error` (K8 review; Rust cannot catch it), which a
//!   worker would hit again on every start until the cache is cleared. A file that does not
//!   match its name (torn write, disk damage, edits) is deleted and rebuilt instead.
//! - **Use**: every PSO is first created with the archive attached and
//!   `MTLPipelineOptionFailOnBinaryArchiveMiss` (a hit loads the stored binary, no compile).
//!   On a miss the function is compiled into the archive (`addComputePipelineFunctions`) and the
//!   PSO is then loaded from it, so a cold kernel is compiled once (creating the PSO the plain way
//!   and then adding it compiled twice whenever the OS shader cache was off: 270 s instead of
//!   24 s for all 71 keccak kernels, K5 note). At the end the archive is serialized (to
//!   a temporary file, then renamed into place).
//! - **Fallback**: any archive error (unsupported device, unreadable/corrupt/foreign file, failed
//!   add or serialize) is logged and the PSOs are created without an archive; a broken file is
//!   deleted so the next process writes a fresh one. The archive never makes a load fail that
//!   would have succeeded without it.
//! - `RISC0_METAL_ARCHIVE=0` disables it.
//!
//! PSOs are created on a few threads at once (the compiler service compiles them in parallel),
//! and those threads add to the one archive concurrently, without a lock: Apple does not document
//! `addComputePipelineFunctions` as thread-safe, so every add is checked by reloading the PSO
//! with `FailOnBinaryArchiveMiss`, and a failed add or reload falls back to plain PSO creation
//! (no failure was seen in the K5 runs). Each worker runs in its own autorelease pool, so the
//! Objective-C objects the Metal calls autorelease (errors, descriptions) are freed per kernel.
//!
//! R20 adds [specialized_pipeline]: one PSO of a kernel specialized with function constants
//! (`MTLFunctionConstantValues`), in an archive of its own whose key includes the constant
//! values (rv32im's per-domain `ef_eval_split_fc`).
//!
//! Z16 adds an optional read-only **offline (AOT) archive** per unit: `RISC0_METAL_AOT_DIR` names a
//! directory of archives made ahead of time by `metal-tt` (`tools/metal_aot/build_archive.py`)
//! for one GPU architecture each, named (since W7-Z1, below)
//! `<unit>-<parallel_digest(metallib)[..24]>-<MTLDevice.architecture.name>-<os key>-<checksum>.binarchive`.
//! The one matching this metallib and this GPU is checked against its checksum, loaded, and tried
//! first for every PSO (`FailOnBinaryArchiveMiss`); a miss continues as above (the runtime
//! archive, then a compile). So a first install on a GPU the shipped set covers compiles
//! nothing, and any other GPU, OS or metallib just misses (a wrong archive is never fatal: Metal
//! reports a miss, Z16 note). The directory is never written. Unset: no change.
//!
//! W7-Z1 completes the offline archives:
//! - **AOT key**: the file name also carries an OS key,
//!   `<unit>-<parallel_digest(metallib)[..24]>-<arch>-<os key>-<checksum>.binarchive`, where the
//!   OS key is `os<ProductBuildVersion>` (e.g. `os26B131`: made for exactly this OS build) or
//!   `macos<major>` (e.g. `macos27`: made with `metal-tt -macos-version-min <major>.0` for every
//!   build of that major version, the packaging form). The runtime takes the exact-build file
//!   first, then the major-version one; a file of another OS key is never opened (a clean miss).
//!   AOT binaries come from the downloaded Metal toolchain (`metal-tt`), not the OS compiler, so
//!   the key cannot name the compiler the OS would use; Metal itself treats a binary it cannot
//!   use as a miss (Z16 note), which then costs a compile, never a wrong PSO. The digest and
//!   checksum of AOT names are always [parallel_digest] ([ARCHIVE_HASH_ENV] only switches the
//!   runtime archive's), so they match `tools/metal_aot/build_archive.py` in every hash mode.
//! - **Who reads [AOT_DIR_ENV]**: only process-wide loads, i.e. those whose runtime archive
//!   directory is the process's [archive_dir] ([process_aot_dir]). The explicit-directory test
//!   seams ([ArchivedKernels::new_in] with a directory of their own) never see an offline
//!   archive; [ArchivedKernels::new_in_aot] passes one explicitly.
//! - **Seeding**: when the offline archive served every PSO and there was no runtime archive,
//!   the offline file is copied (no compile) into the runtime archive on a background thread
//!   ([seed_runtime_archive]), so a later start without the offline directory is still warm.
//!   `RISC0_METAL_AOT_SEED=0` disables it.
//! - **Specialized PSOs** ([specialized_pipeline]) try the offline archive of their unit too
//!   (`build_archive.py --fc`), unless `RISC0_METAL_AOT_FC=0`.
//! - **GC** ([gc_archives], [start_gc]): at most once a day per directory, on background
//!   threads after each process-wide load has opened its archive. Every unit this process loads
//!   keeps its current key (registered before it is opened, [register_live]) and its
//!   `[DEFAULT_GC_KEEP] - 1` most recently used other keys; units it does not load lose only keys
//!   unused for 30 days; a file used or written in the last hour is never deleted; stale
//!   build-cache metallibs beside the archive directory go after a day.
//!   `RISC0_METAL_ARCHIVE_GC=0` disables it, `=dry` only logs. Test suites and dev runs on a
//!   shared machine should set `RISC0_METAL_ARCHIVE_GC=0` when they use the default directory:
//!   their GC acts on everyone's archives.
//! - **Warm loads** never open the offline archive: when the runtime archive file was loaded,
//!   each PSO is looked up there first and the offline file is read, checksummed and opened only
//!   at the first miss.
//!
//! The functions marked `#[doc(hidden)]` are test seams (keccak's archive tests use them).

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Mutex,
    },
    time::{Duration, Instant, SystemTime},
};

use crate::core::hash::sha::{cpu::Impl as Sha256Impl, Sha256 as _};
use anyhow::{anyhow, bail, Result};
use metal::{
    foreign_types::ForeignType,
    objc::{msg_send, runtime::Object, sel, sel_impl},
    BinaryArchive, BinaryArchiveDescriptor, BinaryArchiveRef, ComputePipelineDescriptor,
    ComputePipelineDescriptorRef, ComputePipelineState, DeviceRef, Library, MTLPipelineOption, URL,
};

/// Disables the binary archive when `0`.
pub const ARCHIVE_ENV: &str = "RISC0_METAL_ARCHIVE";
/// Overrides the archive directory.
pub const ARCHIVE_DIR_ENV: &str = "RISC0_METAL_ARCHIVE_DIR";
/// Z16: directory of read-only offline (`metal-tt`) archives (module docs). Unset: none.
pub const AOT_DIR_ENV: &str = "RISC0_METAL_AOT_DIR";
/// W7-Z1: `0` keeps the specialized (function-constant) PSOs out of the offline archives.
pub const AOT_FC_ENV: &str = "RISC0_METAL_AOT_FC";
/// W7-Z1: `0` disables copying a fully used offline archive into the runtime archive.
pub const AOT_SEED_ENV: &str = "RISC0_METAL_AOT_SEED";
/// W7-Z1: `0` disables the archive GC ([gc_archives]); `dry` logs what it would delete.
pub const ARCHIVE_GC_ENV: &str = "RISC0_METAL_ARCHIVE_GC";
/// W7-Z1: how many keys per unit the GC keeps (default [DEFAULT_GC_KEEP], minimum 1).
pub const ARCHIVE_GC_KEEP_ENV: &str = "RISC0_METAL_ARCHIVE_GC_KEEP";
const ARCHIVE_EXT: &str = "binarchive";

/// What happened to the archive while the PSOs were built.
#[derive(Clone, Debug, Default)]
pub struct ArchiveReport {
    /// The archive file (None: disabled or no cache directory).
    pub path: Option<PathBuf>,
    /// An existing archive file was loaded.
    pub loaded: bool,
    /// PSOs loaded from the archive (no GPU compile), including [Self::aot_hits].
    pub hits: usize,
    /// Z16: of [Self::hits], PSOs loaded from the offline archive ([AOT_DIR_ENV]).
    pub aot_hits: usize,
    /// Z16: the offline archive file that was loaded, if any.
    pub aot_path: Option<PathBuf>,
    /// W7-Z1: size of [Self::aot_path] in bytes (mapped by Metal while the archive is open).
    pub aot_bytes: u64,
    /// W7-Z1: a background copy of the offline archive into the runtime archive was started
    /// ([seed_runtime_archive]).
    pub seeding: bool,
    /// PSOs not in the archive: compiled into it (or, on an archive error, built the plain way).
    /// Either compiles, unless the OS shader cache has the function.
    pub misses: usize,
    /// The archive was written (it gained functions).
    pub saved: bool,
    /// Archive errors that made us fall back (logged as warnings).
    pub errors: Vec<String>,
    /// Wall time of all PSO creation (including archive load/add/serialize).
    pub elapsed: Duration,
    /// Sum over threads of the time spent adding missed functions to the archive (compiling).
    pub add_time: Duration,
    /// Writing the archive file.
    pub save_time: Duration,
}

/// A metallib with a PSO for each requested kernel, built through the binary archive.
pub struct ArchivedKernels {
    /// Kept for the lifetime of its PSOs (as `MetalKernels::library`).
    library: Library,
    pipelines: HashMap<String, ComputePipelineState>,
    pub report: ArchiveReport,
}

impl ArchivedKernels {
    /// The loaded metallib (e.g. to build a PSO that was not requested, on first use).
    pub fn library(&self) -> &metal::LibraryRef {
        &self.library
    }

    /// The PSO for `name`; panics (a programming error, plan 2.9) if it was not requested.
    pub fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.get(name)
            .unwrap_or_else(|| panic!("Metal kernel {name} was not loaded"))
    }

    pub fn get(&self, name: &str) -> Option<&ComputePipelineState> {
        self.pipelines.get(name)
    }

    /// The library, the PSOs by kernel name and the archive report (Z16: the zkp and recursion
    /// loaders keep their own containers).
    pub fn into_parts(
        self,
    ) -> (
        Library,
        HashMap<String, ComputePipelineState>,
        ArchiveReport,
    ) {
        (self.library, self.pipelines, self.report)
    }

    /// Load `bytes` as a metallib and build a PSO for each of `names` (through the archive of
    /// `unit`, see the module docs). Fails on a missing kernel or a PSO the device rejects; never
    /// because of the archive.
    pub fn new(device: &DeviceRef, unit: &str, bytes: &[u8], names: &[&str]) -> Result<Self> {
        Self::new_in(archive_dir(), device, unit, bytes, names)
    }

    /// [Self::new] with the archive in `dir` (None: no archive).
    pub fn new_in(
        dir: Option<PathBuf>,
        device: &DeviceRef,
        unit: &str,
        bytes: &[u8],
        names: &[&str],
    ) -> Result<Self> {
        Self::new_in_threads(dir, device, unit, bytes, names, None)
    }

    /// [Self::new_in] on at most `threads` PSO-building threads (None: `available_parallelism`;
    /// Z16: recursion's `RISC0_METAL_REC_PSO_THREADS`).
    ///
    /// W7-Z1: the offline archive ([AOT_DIR_ENV]) is used only when `dir` is the process's
    /// archive directory ([process_aot_dir]): the process-wide loaders pass [archive_dir], the
    /// tests their own directories, which therefore never see an offline archive from the
    /// environment (PLAN 8.7).
    pub fn new_in_threads(
        dir: Option<PathBuf>,
        device: &DeviceRef,
        unit: &str,
        bytes: &[u8],
        names: &[&str],
        threads: Option<usize>,
    ) -> Result<Self> {
        let aot = process_aot_dir(dir.as_deref());
        Self::new_in_aot(dir, aot, device, unit, bytes, names, threads)
    }

    /// [Self::new_in_threads] with the offline archive directory given (None: no offline
    /// archive) instead of read from [AOT_DIR_ENV] (tests never set process-wide environment).
    #[doc(hidden)]
    pub fn new_in_aot(
        dir: Option<PathBuf>,
        aot_dir: Option<PathBuf>,
        device: &DeviceRef,
        unit: &str,
        bytes: &[u8],
        names: &[&str],
        threads: Option<usize>,
    ) -> Result<Self> {
        let t0 = Instant::now();
        // A relative directory (e.g. `RISC0_METAL_ARCHIVE_DIR=rel`) would become a `file://rel/..`
        // URL that `serializeToURL` cannot open, so nothing would ever be saved: resolve it
        // against the current directory first.
        let dir = dir.map(|d| std::path::absolute(&d).unwrap_or(d));
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
            bail!("metallib is missing kernel(s): {}", missing.join(", "));
        }

        let base = dir
            .as_deref()
            .and_then(|dir| archive_base(dir, device, unit, bytes, names));
        let mut report = ArchiveReport {
            path: base.as_deref().map(|b| b.with_extension(ARCHIVE_EXT)),
            ..Default::default()
        };
        if dir.is_some() && base.is_none() {
            report
                .errors
                .push("OS build unknown: binary archive not used".into());
        }
        if let Some(base) = base.as_deref() {
            register_live(base);
        }
        let opened = SystemTime::now();
        let archive = base
            .as_deref()
            .and_then(|base| open_archive(device, base, &mut report));
        if dir.as_deref().is_some_and(is_process_dir) {
            start_gc(dir.as_deref().unwrap(), unit);
        }
        let aot = LazyAot::new(aot_dir, device, unit, bytes);

        let add_nanos = AtomicU64::new(0);
        let built = build_all(
            device,
            &library,
            names,
            Archives {
                aot: &aot,
                runtime: archive.as_deref(),
                runtime_first: report.loaded,
            },
            &add_nanos,
            threads,
        )?;
        aot.merge_into(&mut report);
        let mut pipelines = HashMap::with_capacity(names.len());
        let mut added = 0;
        for (name, pso, source) in built {
            match source {
                Source::Hit => report.hits += 1,
                Source::AotHit => {
                    report.hits += 1;
                    report.aot_hits += 1;
                }
                Source::Added => {
                    report.misses += 1;
                    added += 1;
                }
                Source::Plain(err) => {
                    report.misses += 1;
                    if let Some(err) = err {
                        let msg = format!("{name}: {err}");
                        tracing::warn!("{unit}: Metal binary archive: {msg}; built without it");
                        report.errors.push(msg);
                    }
                }
            }
            super::runtime::register_pso_name(&pso, name);
            pipelines.insert(name.to_string(), pso);
        }
        report.add_time = Duration::from_nanos(add_nanos.load(Ordering::Relaxed));
        maybe_seed(
            &mut report,
            base.as_deref(),
            unit,
            names.len(),
            archive.is_some(),
        );

        if let (Some(archive), Some(base)) = (archive.as_deref(), base.as_deref()) {
            if added > 0 {
                let t_save = Instant::now();
                match save_archive(archive, base, unit, opened) {
                    Ok(path) => {
                        report.saved = true;
                        report.path = Some(path);
                    }
                    Err(e) => {
                        let msg = format!("saving {}: {e:#}", base.display());
                        tracing::warn!("{unit}: Metal binary archive: {msg}; PSOs unaffected");
                        report.errors.push(msg);
                    }
                }
                report.save_time = t_save.elapsed();
            }
        }
        report.elapsed = t0.elapsed();
        tracing::debug!(
            "{unit}: {} PSOs in {:.2} s (archive {}: {} hits, {} misses, loaded {}, adds {:.2} s, saved {} in {:.2} s; offline {}: {} hits)",
            names.len(),
            report.elapsed.as_secs_f64(),
            report
                .path
                .as_deref()
                .map_or("off".into(), |p| p.display().to_string()),
            report.hits - report.aot_hits,
            report.misses,
            report.loaded,
            report.add_time.as_secs_f64(),
            report.saved,
            report.save_time.as_secs_f64(),
            report
                .aot_path
                .as_deref()
                .map_or("not opened".into(), |p| p.display().to_string()),
            report.aot_hits,
        );
        Ok(Self {
            library,
            pipelines,
            report,
        })
    }
}

/// The archive directory: [ARCHIVE_DIR_ENV], else `<risc0 cache dir>/metal-archives`; None when
/// the archive is disabled ([ARCHIVE_ENV] = 0) or there is no home directory.
pub fn archive_dir() -> Option<PathBuf> {
    if std::env::var(ARCHIVE_ENV).is_ok_and(|v| v.trim() == "0") {
        return None;
    }
    if let Some(dir) = std::env::var_os(ARCHIVE_DIR_ENV).filter(|d| !d.is_empty()) {
        return Some(dir.into());
    }
    // risc0-build-kernel's `risc0_cache()`: `directories::ProjectDirs::from("com.risczero",
    // "RISC Zero", "risc0").cache_dir()`, which is this path on macOS.
    let home = std::env::var_os("HOME").filter(|h| !h.is_empty())?;
    Some(
        Path::new(&home)
            .join("Library/Caches/com.risczero.RISC-Zero.risc0")
            .join("metal-archives"),
    )
}

/// Z16: the offline archive directory ([AOT_DIR_ENV]); None when unset/empty or when the
/// archive is disabled ([ARCHIVE_ENV] = 0).
pub fn aot_dir() -> Option<PathBuf> {
    if std::env::var(ARCHIVE_ENV).is_ok_and(|v| v.trim() == "0") {
        return None;
    }
    std::env::var_os(AOT_DIR_ENV)
        .filter(|d| !d.is_empty())
        .map(|d| {
            let d = PathBuf::from(d);
            std::path::absolute(&d).unwrap_or(d)
        })
}

/// Whether `dir` is the process's runtime archive directory ([archive_dir]), compared as
/// absolute paths: the mark of a process-wide load ([process_aot_dir], [start_gc]).
#[doc(hidden)]
pub fn is_process_dir(dir: &Path) -> bool {
    let abs = |d: &Path| std::path::absolute(d).unwrap_or_else(|_| d.to_path_buf());
    archive_dir().is_some_and(|p| abs(&p) == abs(dir))
}

/// W7-Z1: the offline archive directory for a load whose runtime archive is in `dir`:
/// [aot_dir] when `dir` is the process's [archive_dir] (a process-wide load), None otherwise
/// (a test or tool with a directory of its own never reads [AOT_DIR_ENV]).
pub fn process_aot_dir(dir: Option<&Path>) -> Option<PathBuf> {
    dir.filter(|d| is_process_dir(d)).and_then(|_| aot_dir())
}

/// Z16: `MTLDevice.architecture.name` (e.g. `applegpu_g17s`, the `metal-tt -arch` name); None if
/// the OS does not have it (macOS < 14).
#[allow(unexpected_cfgs)]
pub fn gpu_architecture(device: &DeviceRef) -> Option<String> {
    unsafe {
        let responds: bool = msg_send![device, respondsToSelector: sel!(architecture)];
        if !responds {
            return None;
        }
        let arch: *mut Object = msg_send![device, architecture];
        if arch.is_null() {
            return None;
        }
        let name: *mut Object = msg_send![arch, name];
        if name.is_null() {
            return None;
        }
        let c: *const std::ffi::c_char = msg_send![name, UTF8String];
        (!c.is_null()).then(|| std::ffi::CStr::from_ptr(c).to_string_lossy().into_owned())
    }
}

/// W7-Z1: the OS keys an offline archive may carry on this machine, in the order they are
/// tried: `os<ProductBuildVersion>` (made for exactly this OS build), then `macos<major>`
/// (made for every build of this major version). Empty when neither is known.
pub fn aot_os_keys() -> Vec<String> {
    aot_os_keys_for(os_build().as_deref(), os_version().as_deref())
}

/// [aot_os_keys] for the given OS build and product version (`27.2.1`).
#[doc(hidden)]
pub fn aot_os_keys_for(build: Option<&str>, version: Option<&str>) -> Vec<String> {
    let mut keys = Vec::new();
    if let Some(b) = build.filter(|b| is_key_part(b)) {
        keys.push(format!("os{b}"));
    }
    let major = version
        .and_then(|v| v.split('.').next())
        .filter(|m| !m.is_empty() && m.bytes().all(|c| c.is_ascii_digit()));
    if let Some(m) = major {
        keys.push(format!("macos{m}"));
    }
    keys
}

/// An OS key part must not contain the `-` that separates the name's fields.
fn is_key_part(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|c| c.is_ascii_alphanumeric())
}

/// Z16 / W7-Z1: `<unit>-<parallel_digest(bytes)[..24]>-<arch>-<os_key>`, the stem of the offline
/// archive of a metallib for one GPU architecture and OS key ([aot_os_keys]);
/// `<stem>-<aot_checksum>.binarchive` is the file. Always [parallel_digest], whatever
/// [ARCHIVE_HASH_ENV] says, because `tools/metal_aot/build_archive.py` writes these names.
#[doc(hidden)]
pub fn aot_stem(unit: &str, bytes: &[u8], arch: &str, os_key: &str) -> String {
    aot_stem_of(unit, &parallel_digest(bytes), arch, os_key)
}

/// [aot_stem] with the metallib digest already computed (hex).
fn aot_stem_of(unit: &str, digest: &str, arch: &str, os_key: &str) -> String {
    format!("{unit}-{}-{arch}-{os_key}", &digest[..24])
}

/// The checksum of an offline archive file's content: the first [CHECKSUM_LEN] hex digits of its
/// [parallel_digest] (as `build_archive.py`; independent of [ARCHIVE_HASH_ENV]).
#[doc(hidden)]
pub fn aot_checksum(content: &[u8]) -> String {
    parallel_digest(content)[..CHECKSUM_LEN].to_string()
}

/// The file name of an offline archive with stem `stem` ([aot_stem]) and content `content`.
#[doc(hidden)]
pub fn aot_file_name(stem: &str, content: &[u8]) -> String {
    format!("{stem}-{}.{ARCHIVE_EXT}", aot_checksum(content))
}

/// Z16: load the offline archive of `unit`/`bytes` for this GPU and OS from `dir`, if there is
/// one whose content matches the checksum in its name (a damaged archive can abort the process
/// inside Metal, K8; the module docs). The OS keys are tried in [aot_os_keys] order; a file of
/// another architecture, metallib or OS key is not opened at all. Never fatal and never writes
/// `dir`: a missing, mismatched or unloadable file is reported (debug/warn) and skipped.
fn open_aot_archive(
    dir: &Path,
    device: &DeviceRef,
    unit: &str,
    bytes: &[u8],
    report: &mut ArchiveReport,
) -> Option<BinaryArchive> {
    let arch = gpu_architecture(device)?;
    open_aot_archive_for(dir, device, unit, bytes, &arch, &aot_os_keys(), report)
}

/// [open_aot_archive] with the architecture and OS keys given (tests).
#[doc(hidden)]
pub fn open_aot_archive_for(
    dir: &Path,
    device: &DeviceRef,
    unit: &str,
    bytes: &[u8],
    arch: &str,
    os_keys: &[String],
    report: &mut ArchiveReport,
) -> Option<BinaryArchive> {
    let names: Vec<String> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|e| e.file_name().to_str().map(str::to_owned))
        .filter(|n| n.starts_with(unit) && n.ends_with(ARCHIVE_EXT))
        .collect();
    if names.is_empty() {
        tracing::debug!("{unit}: no offline archive in {}", dir.display());
        return None;
    }
    let digest = parallel_digest(bytes);
    for os_key in os_keys {
        let stem = aot_stem_of(unit, &digest, arch, os_key);
        for name in &names {
            let Some(sum) = checksum_in_name(&stem, name) else {
                continue;
            };
            let path = dir.join(name);
            let content = std::fs::read(&path);
            let ok = content.as_ref().is_ok_and(|c| aot_checksum(c) == sum);
            if !ok {
                let msg = format!(
                    "offline archive {}: checksum mismatch; skipped",
                    path.display()
                );
                tracing::warn!("{unit}: Metal binary archive: {msg}");
                report.errors.push(msg);
                continue;
            }
            let size = content.map_or(0, |c| c.len() as u64);
            let desc = BinaryArchiveDescriptor::new();
            desc.set_url(&file_url(&path));
            match device.new_binary_archive_with_descriptor(&desc) {
                Ok(a) => {
                    report.aot_path = Some(path);
                    report.aot_bytes = size;
                    return Some(a);
                }
                Err(e) => {
                    let msg = format!("offline archive {}: {e}; skipped", path.display());
                    tracing::warn!("{unit}: Metal binary archive: {msg}");
                    report.errors.push(msg);
                }
            }
        }
    }
    tracing::debug!(
        "{unit}: no offline archive {unit}-{}-{arch}-<{}>-*.{ARCHIVE_EXT} in {}",
        &digest[..24],
        os_keys.join("|"),
        dir.display()
    );
    None
}

/// W7-Z1: copy the offline archive `aot` into the runtime archive of `base` ([archive_base]) as
/// [save_archive] would write it (temporary file, checksum name, rename; then the same cleanup),
/// unless a runtime archive file of `base` exists already. Every PSO of the load that calls it
/// was an offline-archive hit, so the copy holds them all for this device and OS: a later start
/// without [AOT_DIR_ENV] (or after the offline directory went away) loads them without a
/// compile. Returns the file written (None: one existed).
#[doc(hidden)]
pub fn seed_runtime_archive(aot: &Path, base: &Path, unit: &str) -> Result<Option<PathBuf>> {
    let dir = base
        .parent()
        .ok_or_else(|| anyhow!("no parent directory"))?;
    if archive_used(base).is_some() {
        return Ok(None);
    }
    std::fs::create_dir_all(dir)?;
    // The runtime checksum is the offline one (both the parallel digest) unless
    // [ARCHIVE_HASH_ENV] switched the runtime's: then the content is hashed again.
    let sum_in_name = aot
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_suffix(&format!(".{ARCHIVE_EXT}")))
        .and_then(|n| n.rsplit_once('-'))
        .map(|(_, sum)| sum.to_string());
    let sum = match sum_in_name {
        Some(sum) if !std::env::var(ARCHIVE_HASH_ENV).is_ok_and(|v| v == "sha256") => sum,
        _ => archive_checksum(&std::fs::read(aot)?),
    };
    let mut name = base.file_name().unwrap().to_os_string();
    name.push(format!("-{sum}.{ARCHIVE_EXT}"));
    let path = base.with_file_name(name);
    let tmp = dir.join(format!(
        ".{}.{ARCHIVE_EXT}.tmp{}",
        base.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    // On APFS `fs::copy` clones the file (no data copied, no extra space until one changes).
    std::fs::copy(aot, &tmp)?;
    // A clone keeps the offline file's time; the GC reads it as "last used".
    std::fs::File::options()
        .write(true)
        .open(&tmp)?
        .set_modified(SystemTime::now())?;
    std::fs::rename(&tmp, &path)?;
    let stem = base.file_name().unwrap().to_string_lossy().into_owned();
    prune_other_keys(dir, unit, &stem, keep_archives(), SystemTime::now());
    Ok(Some(path))
}

/// Whether the offline-archive seeding is enabled ([AOT_SEED_ENV] is not `0`).
fn seed_enabled() -> bool {
    !std::env::var(AOT_SEED_ENV).is_ok_and(|v| v.trim() == "0")
}

/// Start [seed_runtime_archive] on a background thread when the offline archive served all
/// `n` PSOs of a load and no runtime archive was loaded (the runtime archive exists as an
/// object, so `base` is writable in principle).
fn maybe_seed(
    report: &mut ArchiveReport,
    base: Option<&Path>,
    unit: &str,
    n: usize,
    have_runtime: bool,
) {
    let (Some(aot), Some(base)) = (report.aot_path.clone(), base) else {
        return;
    };
    if n == 0 || report.aot_hits != n || report.loaded || !have_runtime || !seed_enabled() {
        return;
    }
    let base = base.to_path_buf();
    let unit = unit.to_string();
    let spawned = std::thread::Builder::new()
        .name(format!("metal-aot-seed-{unit}"))
        .spawn(move || {
            let t = Instant::now();
            match seed_runtime_archive(&aot, &base, &unit) {
                Ok(Some(p)) => tracing::debug!(
                    "{unit}: offline archive copied to {} in {:.3} s",
                    p.display(),
                    t.elapsed().as_secs_f64()
                ),
                Ok(None) => {}
                Err(e) => tracing::warn!(
                    "{unit}: copying the offline archive {} into the runtime archive: {e:#}",
                    aot.display()
                ),
            }
        });
    report.seeding = spawned.is_ok();
}

/// The OS build (`ProductBuildVersion`, e.g. `25A354`): a new build can bring a new GPU compiler,
/// whose binaries the old archive does not hold. None when neither the plist nor `sysctl` gives
/// it: the archive is then not used at all ([archive_base]), because a key without the OS build
/// would survive an OS update (K8).
///
/// W7-Z1: read once per process (the OS cannot change under a running process; every unit and
/// specialization asked for it, one plist read each).
#[doc(hidden)]
pub fn os_build() -> Option<String> {
    static B: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    B.get_or_init(|| {
        plist_value("ProductBuildVersion").or_else(|| {
            std::process::Command::new("/usr/sbin/sysctl")
                .args(["-n", "kern.osversion"])
                .output()
                .ok()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .filter(|s| !s.is_empty())
        })
    })
    .clone()
}

/// W7-Z1: the OS product version (`ProductVersion`, e.g. `27.2.1`), read once per process.
#[doc(hidden)]
pub fn os_version() -> Option<String> {
    static V: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    V.get_or_init(|| plist_value("ProductVersion")).clone()
}

/// A string value of `/System/Library/CoreServices/SystemVersion.plist`.
fn plist_value(key: &str) -> Option<String> {
    let plist = std::fs::read_to_string("/System/Library/CoreServices/SystemVersion.plist").ok()?;
    plist
        .split(&format!("<key>{key}</key>"))
        .nth(1)
        .and_then(|rest| rest.split("<string>").nth(1))
        .and_then(|rest| rest.split("</string>").next())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Chunk size of [chunked_digest].
#[doc(hidden)]
pub const DIGEST_CHUNK: usize = 1 << 20;

/// Environment switch back to K8's single-stream sha256 for the key and checksum (K7 revert knob):
/// `RISC0_METAL_ARCHIVE_HASH=sha256`. Archives written under one setting have another file name
/// (key) and checksum than under the other, so switching costs one cold build.
pub const ARCHIVE_HASH_ENV: &str = "RISC0_METAL_ARCHIVE_HASH";

/// The hex digest the archive key and checksum are built from (K7): sha256 over the sha256s of
/// the 1 MiB chunks of `bytes`, the chunks hashed on all cores (K8 finding 5: hashing the
/// 16 MB metallib and the ~50 MB archive in one stream took ~130 ms of every warm start).
/// Any changed byte changes its chunk's digest and so the result, like a plain sha256 (a Merkle
/// list of one level). With [ARCHIVE_HASH_ENV]=sha256: plain sha256 of `bytes`.
#[doc(hidden)]
pub fn chunked_digest(bytes: &[u8]) -> String {
    if std::env::var(ARCHIVE_HASH_ENV).is_ok_and(|v| v == "sha256") {
        return Sha256Impl::hash_bytes(bytes).to_string();
    }
    parallel_digest(bytes)
}

/// [chunked_digest] without the [ARCHIVE_HASH_ENV] switch: always the parallel chunked form.
#[doc(hidden)]
pub fn parallel_digest(bytes: &[u8]) -> String {
    let chunks: Vec<&[u8]> = bytes.chunks(DIGEST_CHUNK).collect();
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, chunks.len().max(1));
    let per = chunks.len().div_ceil(threads).max(1);
    let mut list = Vec::with_capacity(8 + chunks.len() * 32);
    list.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    std::thread::scope(|scope| {
        let workers: Vec<_> = chunks
            .chunks(per)
            .map(|group| {
                scope.spawn(move || {
                    group
                        .iter()
                        .map(|c| Sha256Impl::hash_bytes(c).as_bytes().to_vec())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for w in workers {
            for d in w.join().expect("archive digest worker") {
                list.extend_from_slice(&d);
            }
        }
    });
    Sha256Impl::hash_bytes(&list).to_string()
}

/// The archive key: metallib, device, OS build and the kernel names.
#[doc(hidden)]
pub fn archive_key(device_name: &str, os: &str, bytes: &[u8], names: &[&str]) -> String {
    let mut input = chunked_digest(bytes).into_bytes();
    for part in [device_name, os].into_iter().chain(names.iter().copied()) {
        input.push(0);
        input.extend_from_slice(part.as_bytes());
    }
    let digest = Sha256Impl::hash_bytes(&input).to_string();
    digest[..24].to_string()
}

/// `<dir>/<unit>-<key>`: the archive file is this plus `-<checksum>.binarchive`
/// ([archive_file]). None (no archive) when the OS build is unknown ([os_build]).
#[doc(hidden)]
pub fn archive_base(
    dir: &Path,
    device: &DeviceRef,
    unit: &str,
    bytes: &[u8],
    names: &[&str],
) -> Option<PathBuf> {
    archive_base_for(
        dir,
        device.name(),
        os_build().as_deref(),
        unit,
        bytes,
        names,
    )
}

/// [archive_base] with the device name and OS build given.
#[doc(hidden)]
pub fn archive_base_for(
    dir: &Path,
    device_name: &str,
    os: Option<&str>,
    unit: &str,
    bytes: &[u8],
    names: &[&str],
) -> Option<PathBuf> {
    let key = archive_key(device_name, os?, bytes, names);
    Some(dir.join(format!("{unit}-{key}")))
}

/// Hex digits of the content checksum in an archive's file name.
const CHECKSUM_LEN: usize = 16;

/// The first [CHECKSUM_LEN] hex digits of the [chunked_digest] of an archive's bytes.
#[doc(hidden)]
pub fn archive_checksum(content: &[u8]) -> String {
    chunked_digest(content)[..CHECKSUM_LEN].to_string()
}

/// The file name of an archive with base `base` ([archive_base]) and content `content`.
#[doc(hidden)]
pub fn archive_file(base: &Path, content: &[u8]) -> PathBuf {
    let mut name = base.file_name().unwrap_or_default().to_os_string();
    name.push(format!("-{}.{ARCHIVE_EXT}", archive_checksum(content)));
    base.with_file_name(name)
}

/// The checksum part of `name` if it is an archive file name of `base`'s stem
/// (`<stem>-<checksum>.binarchive`).
fn checksum_in_name<'a>(stem: &str, name: &'a str) -> Option<&'a str> {
    let sum = name
        .strip_prefix(stem)?
        .strip_prefix('-')?
        .strip_suffix(ARCHIVE_EXT)?
        .strip_suffix('.')?;
    (sum.len() == CHECKSUM_LEN && sum.bytes().all(|b| b.is_ascii_hexdigit())).then_some(sum)
}

/// A `file://` URL for `path` (percent-encoding everything but unreserved characters and `/`).
#[allow(unexpected_cfgs)]
#[doc(hidden)]
pub fn file_url(path: &Path) -> URL {
    let mut s = String::from("file://");
    for b in path.to_string_lossy().bytes() {
        if b.is_ascii_alphanumeric() || b"/-_.~".contains(&b) {
            s.push(b as char);
        } else {
            s.push_str(&format!("%{b:02X}"));
        }
    }
    let url = URL::new_with_string(&s);
    // metal-rs 0.29 wraps `+[NSURL URLWithString:]` (an autoreleased, +0 object) as an owned
    // `URL` that releases on drop: take the reference it will give back.
    unsafe {
        let () = msg_send![url.as_ptr(), retain];
    }
    url
}

/// The archive file of `base` whose content matches the checksum in its name, if any. Files of
/// `base` that do not match (damaged, truncated, edited) are deleted and reported: Metal can
/// abort the process on a damaged archive (module doc), so it never sees one.
fn verified_archive_file(base: &Path, report: &mut ArchiveReport) -> Option<PathBuf> {
    let dir = base.parent()?;
    let stem = base.file_name()?.to_str()?;
    let mut found = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name();
        let Some(sum) = name.to_str().and_then(|n| checksum_in_name(stem, n)) else {
            continue;
        };
        let path = entry.path();
        if found.is_none() {
            match std::fs::read(&path) {
                Ok(content) if archive_checksum(&content) == sum => {
                    found = Some(path);
                    continue;
                }
                Ok(_) => {
                    let msg = format!(
                        "{}: content does not match its checksum; deleting it",
                        path.display()
                    );
                    tracing::warn!("Metal binary archive: {msg}");
                    report.errors.push(msg);
                }
                // Gone meanwhile (another process replaced it): not an error.
                Err(_) => continue,
            }
        }
        // A damaged file, or a second one (only one survives a save): remove it.
        let _ = std::fs::remove_file(&path);
    }
    found
}

/// The archive to build with: the verified file of `base` if it loads, else an empty one; None
/// if even an empty archive cannot be made (e.g. a device without archive support).
fn open_archive(
    device: &DeviceRef,
    base: &Path,
    report: &mut ArchiveReport,
) -> Option<BinaryArchive> {
    open_archive_with(device, base, report, &mut |_| {})
}

/// [open_archive], calling `before_load` between verifying a file and Metal opening it (a test
/// seam for the race below).
///
/// The cache directory is shared by every process (and every checkout) on the machine. Between
/// [verified_archive_file] and Metal's open, another process's [save_archive] can rename its own
/// archive into place and delete the file just verified; Metal then fails ("Invalid URL"). A
/// file that is gone is not damaged: no error, nothing to delete, and the newer file is picked
/// up by verifying once more (one retry: a second loss falls back to an empty archive, which
/// costs a rebuild, never a wrong PSO).
#[doc(hidden)]
pub fn open_archive_with(
    device: &DeviceRef,
    base: &Path,
    report: &mut ArchiveReport,
    before_load: &mut dyn FnMut(&Path),
) -> Option<BinaryArchive> {
    for _attempt in 0..2 {
        let Some(path) = verified_archive_file(base, report) else {
            break;
        };
        before_load(&path);
        let desc = BinaryArchiveDescriptor::new();
        desc.set_url(&file_url(&path));
        match device.new_binary_archive_with_descriptor(&desc) {
            Ok(archive) => {
                report.loaded = true;
                // Mark it used (K7): the stale cleanup of other keys evicts the least recently
                // used archives first.
                if let Ok(f) = std::fs::File::open(&path) {
                    let _ = f.set_modified(SystemTime::now());
                }
                report.path = Some(path);
                return Some(archive);
            }
            Err(e) if !path.exists() => {
                tracing::debug!(
                    "Metal binary archive: {} replaced by another process before loading ({e}); \
                     looking again",
                    path.display()
                );
            }
            Err(e) => {
                let msg = format!("loading {}: {e}; deleting it", path.display());
                tracing::warn!("Metal binary archive: {msg}");
                report.errors.push(msg);
                let _ = std::fs::remove_file(&path);
                break;
            }
        }
    }
    match device.new_binary_archive_with_descriptor(&BinaryArchiveDescriptor::new()) {
        Ok(archive) => Some(archive),
        Err(e) => {
            let msg = format!("creating an empty archive: {e}");
            tracing::warn!("Metal binary archive: {msg}");
            report.errors.push(msg);
            None
        }
    }
}

/// `newComputePipelineStateWithDescriptor:options:reflection:error:` with no reflection
/// (metal-rs 0.29's `new_compute_pipeline_state_with_reflection` wraps a nil reflection).
// `msg_send!` expands a `cfg(feature = "cargo-clippy")` this crate does not declare.
#[allow(unexpected_cfgs)]
#[doc(hidden)]
pub fn new_pso_with_options(
    device: &DeviceRef,
    desc: &ComputePipelineDescriptorRef,
    options: MTLPipelineOption,
) -> std::result::Result<ComputePipelineState, String> {
    unsafe {
        let mut err: *mut Object = std::ptr::null_mut();
        let reflection: *mut Object = std::ptr::null_mut();
        let pso: *mut metal::MTLComputePipelineState = msg_send![device,
            newComputePipelineStateWithDescriptor: desc
            options: options
            reflection: reflection
            error: &mut err];
        if !err.is_null() {
            let desc: *mut Object = msg_send![err, localizedDescription];
            let c_msg: *const std::ffi::c_char = msg_send![desc, UTF8String];
            let message = std::ffi::CStr::from_ptr(c_msg)
                .to_string_lossy()
                .into_owned();
            if !pso.is_null() {
                let () = msg_send![pso, release];
            }
            return Err(message);
        }
        if pso.is_null() {
            return Err("nil pipeline state".into());
        }
        Ok(ComputePipelineState::from_ptr(pso))
    }
}

/// Where a PSO came from.
enum Source {
    /// Loaded from the archive as it was.
    Hit,
    /// Z16: loaded from the offline archive ([AOT_DIR_ENV]).
    AotHit,
    /// Compiled into the archive (`addComputePipelineFunctions`), then loaded from it.
    Added,
    /// Built without the archive: there is none (`None`), or the archive failed (the error).
    Plain(Option<String>),
}

type Built<'a> = (&'a str, ComputePipelineState, Source);

/// The archives a PSO may come from: the read-only offline one (Z16) and the runtime one (hit,
/// else compiled into it). The offline one is tried first unless the runtime archive was loaded
/// from a file ([Self::runtime_first]).
#[derive(Clone, Copy)]
struct Archives<'a> {
    aot: &'a LazyAot<'a>,
    runtime: Option<&'a BinaryArchiveRef>,
    /// The runtime archive was loaded from a file: try it before the offline archive, which is
    /// then opened only if some PSO misses there (fix round 1: a warm start with
    /// [AOT_DIR_ENV] set paid for reading, checksumming and opening every offline file).
    runtime_first: bool,
}

/// W7-Z1: the offline archive of a load, opened ([open_aot_archive]) at its first use by any
/// PSO-building thread, or never.
struct LazyAot<'a> {
    dir: Option<PathBuf>,
    device: &'a DeviceRef,
    unit: &'a str,
    bytes: &'a [u8],
    archive: std::sync::OnceLock<Option<BinaryArchive>>,
    /// What opening it reported ([ArchiveReport::aot_path], `aot_bytes`, errors).
    report: Mutex<ArchiveReport>,
}

impl<'a> LazyAot<'a> {
    fn new(dir: Option<PathBuf>, device: &'a DeviceRef, unit: &'a str, bytes: &'a [u8]) -> Self {
        Self {
            dir,
            device,
            unit,
            bytes,
            archive: Default::default(),
            report: Default::default(),
        }
    }

    fn get(&self) -> Option<&BinaryArchiveRef> {
        let dir = self.dir.as_deref()?;
        self.archive
            .get_or_init(|| {
                let mut r = self.report.lock().unwrap();
                open_aot_archive(dir, self.device, self.unit, self.bytes, &mut r)
            })
            .as_deref()
    }

    fn merge_into(&self, report: &mut ArchiveReport) {
        let mut r = self.report.lock().unwrap();
        report.aot_path = r.aot_path.take();
        report.aot_bytes = r.aot_bytes;
        report.errors.append(&mut r.errors);
    }
}

/// One PSO of `function` (`name` for messages): an archive hit if possible; else the function
/// is compiled into the archive and the PSO loaded from there; else (archive error, or no
/// archive) a plain build. [build_all]'s per-kernel step.
fn build_function(
    device: &DeviceRef,
    function: &metal::FunctionRef,
    name: &str,
    archives: Archives,
    add_nanos: &AtomicU64,
) -> Result<(ComputePipelineState, Source)> {
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(function));
    let hit = |archive: &BinaryArchiveRef| {
        desc.set_binary_archives(&[archive]);
        let pso = new_pso_with_options(device, &desc, MTLPipelineOption::FailOnBinaryArchiveMiss);
        desc.set_binary_archives(&[]);
        pso.ok()
    };
    if let (true, Some(runtime)) = (archives.runtime_first, archives.runtime) {
        if let Some(pso) = hit(runtime) {
            return Ok((pso, Source::Hit));
        }
    }
    if let Some(pso) = archives.aot.get().and_then(hit) {
        return Ok((pso, Source::AotHit));
    }
    let archive = archives.runtime;
    let mut archive_err = None;
    if let Some(archive) = archive {
        desc.set_binary_archives(&[archive]);
        let from_archive =
            || new_pso_with_options(device, &desc, MTLPipelineOption::FailOnBinaryArchiveMiss);
        // (Already tried above when it was loaded from a file.)
        if !archives.runtime_first {
            if let Ok(pso) = from_archive() {
                return Ok((pso, Source::Hit));
            }
        }
        let t = Instant::now();
        let added = archive.add_compute_pipeline_functions_with_descriptor(&desc);
        add_nanos.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        match added {
            Ok(_) => match from_archive() {
                Ok(pso) => return Ok((pso, Source::Added)),
                Err(e) => archive_err = Some(format!("not in the archive after adding: {e}")),
            },
            Err(e) => archive_err = Some(format!("adding to the archive: {e}")),
        }
        desc.set_binary_archives(&[]);
    }
    let pso = device
        .new_compute_pipeline_state(&desc)
        .map_err(|e| anyhow!("compute pipeline for {name}: {e}"))?;
    Ok((pso, Source::Plain(archive_err)))
}

/// The archive-key name of `name` specialized with the `u32` function constants `constants`
/// (`(index, value)` pairs): `name[fc<index>=<value>,...]`.
#[doc(hidden)]
pub fn specialized_name(name: &str, constants: &[(usize, u32)]) -> String {
    let fc: Vec<String> = constants
        .iter()
        .map(|(i, v)| format!("fc{i}={v}"))
        .collect();
    format!("{name}[{}]", fc.join(","))
}

/// R20: the archive base ([archive_base]) of the specialization of `name` with `constants` under
/// `unit`, as [specialized_pipeline] uses it (None when the OS build is unknown).
#[doc(hidden)]
pub fn specialized_archive_base(
    dir: &Path,
    device: &DeviceRef,
    bytes: &[u8],
    unit: &str,
    name: &str,
    constants: &[(usize, u32)],
) -> Option<PathBuf> {
    let dir = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
    archive_base(
        &dir,
        device,
        unit,
        bytes,
        &[specialized_name(name, constants).as_str()],
    )
}

/// R20: the modification time (last written or loaded, see [open_archive_with]) of the newest
/// archive file of `base` ([archive_base]), None if there is none (by name only; the content is
/// verified when it is opened). Lets a caller load only the specializations that a previous
/// process compiled, most recently used first, without compiling the others.
pub fn archive_used(base: &Path) -> Option<SystemTime> {
    let dir = base.parent()?;
    let stem = base.file_name()?.to_str()?;
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| checksum_in_name(stem, n).is_some())
        })
        .filter_map(|e| e.metadata().and_then(|m| m.modified()).ok())
        .max()
}

/// R20: one PSO of the kernel `name` of `library` (loaded from the metallib `bytes`), specialized
/// with the `u32` function constants `constants` (`(index, value)` pairs, through
/// `MTLFunctionConstantValues`), built through its own binary archive in `dir` (None: no
/// archive) under `unit`.
///
/// The archive key ([archive_key]) covers the metallib, the device, the OS build and
/// [specialized_name], so it includes the constant values: each specialization is a separate
/// archive file and unit-level pruning ([prune_other_keys]) works as for [ArchivedKernels].
/// Callers should give each specialization its own `unit` (e.g. with the value in it), so that
/// the specializations of one metallib do not prune each other; unit names follow the module
/// doc's rule (no unit is another's prefix followed by `-`). Same fallbacks as
/// [ArchivedKernels::new_in]: an archive error is reported and logged, never fatal. Blocking
/// (a cold specialization of a large kernel compiles for many seconds), so callers that must not
/// wait run it on a thread of their own.
///
/// W7-Z1: a process-wide call (`dir` is [archive_dir]) first tries the offline archive of
/// `unit` ([AOT_DIR_ENV], `build_archive.py --fc`), unless [AOT_FC_ENV] is `0`. Note that with a
/// cold OS shader cache `newFunctionWithName:constantValues:` still specializes the function in
/// the compiler service before any archive lookup (~6.5 s for rv32im's split kernel, Z16 note):
/// the offline archive removes the GPU compile of a first install, not that step.
pub fn specialized_pipeline(
    dir: Option<PathBuf>,
    device: &DeviceRef,
    library: &metal::LibraryRef,
    bytes: &[u8],
    unit: &str,
    name: &str,
    constants: &[(usize, u32)],
) -> Result<(ComputePipelineState, ArchiveReport)> {
    let aot = if aot_fc_enabled() {
        process_aot_dir(dir.as_deref())
    } else {
        None
    };
    specialized_pipeline_in(dir, aot, device, library, bytes, unit, name, constants)
}

/// Whether specialized PSOs may come from offline archives ([AOT_FC_ENV] is not `0`).
pub fn aot_fc_enabled() -> bool {
    !std::env::var(AOT_FC_ENV).is_ok_and(|v| v.trim() == "0")
}

/// [specialized_pipeline] with the offline archive directory given (None: none) instead of
/// derived from the environment (tests).
#[allow(clippy::too_many_arguments)]
#[doc(hidden)]
pub fn specialized_pipeline_in(
    dir: Option<PathBuf>,
    aot_dir: Option<PathBuf>,
    device: &DeviceRef,
    library: &metal::LibraryRef,
    bytes: &[u8],
    unit: &str,
    name: &str,
    constants: &[(usize, u32)],
) -> Result<(ComputePipelineState, ArchiveReport)> {
    let t0 = Instant::now();
    let dir = dir.map(|d| std::path::absolute(&d).unwrap_or(d));
    let values = metal::FunctionConstantValues::new();
    for (index, value) in constants {
        values.set_constant_value_at_index(
            value as *const u32 as *const std::ffi::c_void,
            metal::MTLDataType::UInt,
            *index as u64,
        );
    }
    let function = library
        .get_function(name, Some(values))
        .map_err(|e| anyhow!("metallib function {name} (specialized {constants:?}): {e}"))?;
    let key_name = specialized_name(name, constants);
    let base = dir
        .as_deref()
        .and_then(|dir| archive_base(dir, device, unit, bytes, &[key_name.as_str()]));
    let mut report = ArchiveReport {
        path: base.as_deref().map(|b| b.with_extension(ARCHIVE_EXT)),
        ..Default::default()
    };
    if dir.is_some() && base.is_none() {
        report
            .errors
            .push("OS build unknown: binary archive not used".into());
    }
    if let Some(base) = base.as_deref() {
        register_live(base);
    }
    let opened = SystemTime::now();
    let archive = base
        .as_deref()
        .and_then(|base| open_archive(device, base, &mut report));
    if dir.as_deref().is_some_and(is_process_dir) {
        start_gc(dir.as_deref().unwrap(), unit);
    }
    let aot = LazyAot::new(aot_dir, device, unit, bytes);
    let add_nanos = AtomicU64::new(0);
    let archives = Archives {
        aot: &aot,
        runtime: archive.as_deref(),
        runtime_first: report.loaded,
    };
    let (pso, source) = metal::objc::rc::autoreleasepool(|| {
        build_function(device, &function, &key_name, archives, &add_nanos)
    })?;
    aot.merge_into(&mut report);
    report.add_time = Duration::from_nanos(add_nanos.load(Ordering::Relaxed));
    let mut added = false;
    match source {
        Source::Hit => report.hits += 1,
        Source::AotHit => {
            report.hits += 1;
            report.aot_hits += 1;
        }
        Source::Added => {
            report.misses += 1;
            added = true;
        }
        Source::Plain(err) => {
            report.misses += 1;
            if let Some(err) = err {
                let msg = format!("{key_name}: {err}");
                tracing::warn!("{unit}: Metal binary archive: {msg}; built without it");
                report.errors.push(msg);
            }
        }
    }
    maybe_seed(&mut report, base.as_deref(), unit, 1, archive.is_some());
    if let (Some(archive), Some(base), true) = (archive.as_deref(), base.as_deref(), added) {
        let t_save = Instant::now();
        match save_archive(archive, base, unit, opened) {
            Ok(path) => {
                report.saved = true;
                report.path = Some(path);
            }
            Err(e) => {
                let msg = format!("saving {}: {e:#}", base.display());
                tracing::warn!("{unit}: Metal binary archive: {msg}; PSO unaffected");
                report.errors.push(msg);
            }
        }
        report.save_time = t_save.elapsed();
    }
    report.elapsed = t0.elapsed();
    Ok((pso, report))
}

/// Each PSO: an archive hit if possible; else its function is compiled into the archive and the
/// PSO loaded from there (one GPU compile per kernel); else (archive error) a plain build. Runs
/// on up to `available_parallelism` threads: the compiler service compiles concurrently.
/// Whether concurrent adds to one `MTLBinaryArchive` are safe is not documented; each add is
/// followed by a `FailOnBinaryArchiveMiss` load that proves the function landed, and a failure
/// falls back to the plain build.
fn build_all<'a>(
    device: &DeviceRef,
    library: &Library,
    names: &[&'a str],
    archives: Archives,
    add_nanos: &AtomicU64,
    threads: Option<usize>,
) -> Result<Vec<Built<'a>>> {
    let build_one = |name: &'a str| -> Result<Built<'a>> {
        let function = library
            .get_function(name, None)
            .map_err(|e| anyhow!("metallib function {name}: {e}"))?;
        let (pso, source) = build_function(device, &function, name, archives, add_nanos)?;
        Ok((name, pso, source))
    };

    let threads = threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
        .min(names.len())
        .max(1);
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<Result<Built<'a>>>>> =
        Mutex::new((0..names.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= names.len() {
                    break;
                }
                // These are not main-thread workers: give each PSO its own autorelease pool.
                let r = metal::objc::rc::autoreleasepool(|| build_one(names[i]));
                results.lock().unwrap()[i] = Some(r);
            });
        }
    });
    results
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|r| r.expect("every PSO built"))
        .collect()
}

/// Write `archive` as the archive file of `base` (temporary file, checksum, rename to
/// [archive_file]), then delete older saves of the same key and prune the unit's other keys ([prune_other_keys]). A file of the
/// same key written at or after `opened` (when this process looked for one) is another process's
/// concurrent save and is kept: deleting it would make the processes overwrite each other's
/// archives in turn. The next [verified_archive_file] keeps one of the two. Returns the file
/// written.
#[doc(hidden)]
pub fn save_archive(
    archive: &BinaryArchiveRef,
    base: &Path,
    unit: &str,
    opened: SystemTime,
) -> Result<PathBuf> {
    let dir = base
        .parent()
        .ok_or_else(|| anyhow!("no parent directory"))?;
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(
        ".{}.{ARCHIVE_EXT}.tmp{}",
        base.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    let _ = std::fs::remove_file(&tmp);
    match archive.serialize_to_url(&file_url(&tmp)) {
        Ok(true) => {}
        Ok(false) => bail!("serializeToURL returned NO"),
        Err(e) => bail!("serializeToURL: {e}"),
    }
    let content = std::fs::read(&tmp)?;
    let path = archive_file(base, &content);
    drop(content);
    std::fs::rename(&tmp, &path)?;
    let keep = path.file_name().unwrap().to_owned();
    let stem = base.file_name().unwrap().to_string_lossy().into_owned();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            let concurrent_save = checksum_in_name(&stem, &s).is_some()
                && entry
                    .metadata()
                    .and_then(|m| m.modified())
                    .is_ok_and(|t| t >= opened);
            // Same key, another file: an older save of this archive (only one survives).
            if name != keep && !concurrent_save && checksum_in_name(&stem, &s).is_some() {
                let _ = std::fs::remove_file(entry.path());
            }
            // The temporary file of a process that died between serialize and rename (~50 MB);
            // only old ones, so another process's write in progress is left alone.
            let old_tmp = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.elapsed().ok())
                .is_some_and(|age| age > Duration::from_secs(3600));
            if s.starts_with(&format!(".{unit}-"))
                && s.contains(&format!(".{ARCHIVE_EXT}.tmp"))
                && old_tmp
            {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    prune_other_keys(dir, unit, &stem, keep_archives(), SystemTime::now());
    Ok(path)
}

/// How many archives of a unit (keys: metallib, device, OS build) the cache dir keeps
/// (`RISC0_METAL_ARCHIVE_KEEP`, default [DEFAULT_ARCHIVE_KEEP]; 1 = K5/K8's rule, only the newest).
pub const ARCHIVE_KEEP_ENV: &str = "RISC0_METAL_ARCHIVE_KEEP";

/// Default of [ARCHIVE_KEEP_ENV]. K8 finding 4: deleting every other key made two binaries with
/// different metallibs on one machine (risc0 versions, worktrees, a worker and a test build)
/// evict each other, so each switch paid a cold build (~25 s on an M5 Max, minutes on base
/// chips). Four keys of ~35-60 MB each bound the directory to a few hundred MB per unit.
pub const DEFAULT_ARCHIVE_KEEP: usize = 4;

/// Archives of other keys unused for this long are deleted regardless of the count (an OS
/// update or a new risc0 version leaves the old ones behind for good).
pub const ARCHIVE_MAX_AGE: Duration = Duration::from_secs(30 * 24 * 3600);

/// The [ARCHIVE_KEEP_ENV] setting.
#[doc(hidden)]
pub fn keep_archives() -> usize {
    std::env::var(ARCHIVE_KEEP_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(DEFAULT_ARCHIVE_KEEP)
}

/// Deletes the unit's archives of keys other than `stem`'s beyond the `keep - 1` most recently
/// used ones (modification time: written or loaded, see [open_archive_with]), and any of them
/// unused for [ARCHIVE_MAX_AGE] (relative to `now`). Never touches `stem`'s own files.
#[doc(hidden)]
pub fn prune_other_keys(dir: &Path, unit: &str, stem: &str, keep: usize, now: SystemTime) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut others: Vec<(SystemTime, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let ours = name.starts_with(&format!("{unit}-")) && name.ends_with(ARCHIVE_EXT);
            if !ours || name.starts_with(&format!("{stem}-")) {
                return None;
            }
            let modified = e.metadata().and_then(|m| m.modified()).ok()?;
            Some((modified, e.path()))
        })
        .collect();
    // Newest first.
    others.sort_by_key(|o| std::cmp::Reverse(o.0));
    for (i, (modified, path)) in others.iter().enumerate() {
        let old = now
            .duration_since(*modified)
            .is_ok_and(|age| age > ARCHIVE_MAX_AGE);
        if i + 1 >= keep || old {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// W7-Z1: keys per unit the GC keeps by default: the current one and the newest previous one.
pub const DEFAULT_GC_KEEP: usize = 2;
/// W7-Z1: the GC runs at most this often (per archive directory, `.gc-stamp`).
pub const GC_INTERVAL: Duration = Duration::from_secs(24 * 3600);
/// W7-Z1: an archive file used (loaded: its modification time is set, [open_archive_with]) or
/// written less than this long ago is never deleted by the GC: a live process may be about to
/// load it, or have it open.
pub const GC_MIN_AGE: Duration = Duration::from_secs(3600);
/// W7-Z1: a build-cache metallib is never deleted within a day of its last use, the rule
/// `risc0-build-kernel`'s own cleanup follows (a build running now may be about to copy it).
pub const GC_METALLIB_MIN_AGE: Duration = Duration::from_secs(24 * 3600);
/// W7-Z1: runtime archive keys of a unit this process has not loaded are deleted once unused for
/// this long ([ARCHIVE_MAX_AGE], the save-time rule).
pub const GC_UNUSED_MAX_AGE: Duration = ARCHIVE_MAX_AGE;

/// What [gc_archives] deleted (or, dry, would delete).
#[derive(Clone, Debug, Default)]
pub struct GcReport {
    /// Files deleted, with their sizes.
    pub deleted: Vec<(PathBuf, u64)>,
    /// Bytes of [Self::deleted].
    pub bytes: u64,
    /// Files of the kept keys.
    pub kept: usize,
    /// Files of evicted keys left alone because they were used too recently.
    pub recent: usize,
}

/// One file the GC considers.
struct GcFile {
    path: PathBuf,
    size: u64,
    modified: SystemTime,
}

/// `<unit>-<key>-<checksum>.binarchive` split into (unit, key) when `key` is 24 and the
/// checksum [CHECKSUM_LEN] hex digits (a runtime archive name; offline names do not match).
fn runtime_archive_parts(name: &str) -> Option<(&str, &str)> {
    let rest = name.strip_suffix(ARCHIVE_EXT)?.strip_suffix('.')?;
    let mut it = rest.rsplitn(3, '-');
    let sum = it.next()?;
    let key = it.next()?;
    let unit = it.next()?;
    let hex = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit());
    (hex(sum, CHECKSUM_LEN) && hex(key, 24) && !unit.is_empty()).then_some((unit, key))
}

/// `<stem>-<key32>.metallib` (`risc0-build-kernel`'s cache names) split into (stem, key);
/// temporary outputs (`.tmp<pid>.metallib`) and bare `<sha256>.metallib` names (one file per
/// unrelated kernel source, no stem to group them by) do not match and are left alone.
fn metallib_parts(name: &str) -> Option<(&str, &str)> {
    let rest = name.strip_suffix(".metallib")?;
    let hex = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_hexdigit());
    match rest.rsplit_once('-') {
        Some((stem, key)) if key.len() == 32 && hex(key) && !stem.is_empty() => Some((stem, key)),
        _ => None,
    }
}

/// How [gc_groups] decides which keys of a group survive.
#[derive(Clone, Copy)]
enum GcPolicy<'a> {
    /// The build cache: the `keep` newest keys of each group.
    Newest(usize),
    /// Runtime archives. A unit with a key that `live(unit, key)` names (this process loaded
    /// or is loading it) keeps those keys and its `keep - 1` newest other keys. A unit with no
    /// live key keeps every key used within `unused` (its current key cannot be known: the
    /// process may load it later, or another binary uses it).
    Live {
        keep: usize,
        unused: Duration,
        live: &'a dyn Fn(&str, &str) -> bool,
    },
}

/// Group `files` by (group, key) and delete the keys `policy` does not keep (a key's time is its
/// newest file's), except files used within `min_age` of `now`.
fn gc_groups(
    files: Vec<(String, String, GcFile)>,
    policy: GcPolicy,
    min_age: Duration,
    now: SystemTime,
    dry: bool,
    report: &mut GcReport,
) {
    let mut groups: HashMap<String, HashMap<String, Vec<GcFile>>> = HashMap::new();
    for (group, key, f) in files {
        groups
            .entry(group)
            .or_default()
            .entry(key)
            .or_default()
            .push(f);
    }
    for (group, keys) in groups {
        let mut keys: Vec<(SystemTime, String, Vec<GcFile>)> = keys
            .into_iter()
            .map(|(k, fs)| (fs.iter().map(|f| f.modified).max().unwrap(), k, fs))
            .collect();
        keys.sort_by_key(|k| std::cmp::Reverse(k.0));
        let any_live = match policy {
            GcPolicy::Live { live, .. } => keys.iter().any(|(_, k, _)| live(&group, k)),
            GcPolicy::Newest(_) => false,
        };
        let mut others = 0;
        for (used, key, fs) in keys {
            let keep_key = match policy {
                GcPolicy::Newest(keep) => {
                    others += 1;
                    others <= keep
                }
                GcPolicy::Live { live, .. } if live(&group, &key) => true,
                GcPolicy::Live { keep, .. } if any_live => {
                    others += 1;
                    others < keep
                }
                GcPolicy::Live { unused, .. } => {
                    now.duration_since(used).map_or(true, |age| age < unused)
                }
            };
            for f in fs {
                if keep_key {
                    report.kept += 1;
                    continue;
                }
                let recent = now
                    .duration_since(f.modified)
                    .map_or(true, |age| age < min_age);
                if recent {
                    report.recent += 1;
                    continue;
                }
                if dry || std::fs::remove_file(&f.path).is_ok() {
                    report.bytes += f.size;
                    report.deleted.push((f.path, f.size));
                }
            }
        }
    }
}

/// W7-Z1: the archive GC. In `dir` (a runtime archive directory), for each unit (keys: metallib,
/// device, OS build, kernel set, [archive_key]):
///
/// - a unit with a key `live(unit, key)` names (this process's own, [register_live]) keeps it
///   and its `keep - 1` most recently used other keys;
/// - a unit with no live key keeps every key used within [GC_UNUSED_MAX_AGE] and loses the rest
///   (a unit nobody uses any more, e.g. a retired kernel set, goes after that time);
/// - a file used (loaded or written) within [GC_MIN_AGE] of `now` is never deleted, nor are
///   offline archives or any other file; temporary files of interrupted saves older than that
///   are.
///
/// In `metallib_dir` (the `risc0-build-kernel` cache, when given) each output stem keeps its
/// `keep` newest metallibs and older ones unused for [GC_METALLIB_MIN_AGE] go. With `dry`
/// nothing is deleted and the report lists what would be.
///
/// Why the current process's keys and not "referenced by the installed workers": a runtime key
/// hashes the metallib digest together with the device, the OS build and the kernel set of each
/// unit (and the function constants of each specialization), which an installed binary does not
/// expose, so the GC cannot compute the keys another binary would use. The one binary whose keys
/// are known is the running one; its other units, and every other binary's, fall under the
/// unused-for rule (fix round 1: the first version kept each unit's two newest keys by time and
/// could delete the running binary's own archives).
pub fn gc_archives(
    dir: &Path,
    metallib_dir: Option<&Path>,
    now: SystemTime,
    keep: usize,
    dry: bool,
    live: &dyn Fn(&str, &str) -> bool,
) -> GcReport {
    gc_archives_of(dir, metallib_dir, now, keep, dry, live, None)
}

/// [gc_archives] restricted to the runtime archives of `only_unit` (when given; temporary files
/// and the build cache are then left to the full pass).
#[doc(hidden)]
pub fn gc_archives_of(
    dir: &Path,
    metallib_dir: Option<&Path>,
    now: SystemTime,
    keep: usize,
    dry: bool,
    live: &dyn Fn(&str, &str) -> bool,
    only_unit: Option<&str>,
) -> GcReport {
    let keep = keep.max(1);
    let mut report = GcReport::default();
    let stat = |e: &std::fs::DirEntry| -> Option<GcFile> {
        let m = e.metadata().ok()?;
        m.is_file().then(|| GcFile {
            path: e.path(),
            size: m.len(),
            modified: m.modified().unwrap_or(now),
        })
    };
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut files = Vec::new();
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let Some(f) = stat(&e) else { continue };
            if let Some((unit, key)) = runtime_archive_parts(&name) {
                if only_unit.is_none_or(|u| u == unit) {
                    files.push((unit.to_string(), key.to_string(), f));
                }
            } else if only_unit.is_none()
                && name.starts_with('.')
                && name.contains(&format!(".{ARCHIVE_EXT}.tmp"))
            {
                let old = now
                    .duration_since(f.modified)
                    .is_ok_and(|age| age >= GC_MIN_AGE);
                if old && (dry || std::fs::remove_file(&f.path).is_ok()) {
                    report.bytes += f.size;
                    report.deleted.push((f.path, f.size));
                }
            }
        }
        let policy = GcPolicy::Live {
            keep,
            unused: GC_UNUSED_MAX_AGE,
            live,
        };
        gc_groups(files, policy, GC_MIN_AGE, now, dry, &mut report);
    }
    if let Some(Ok(entries)) = metallib_dir
        .filter(|_| only_unit.is_none())
        .map(std::fs::read_dir)
    {
        let files = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let (stem, key) = metallib_parts(&name)?;
                Some((stem.to_string(), key.to_string(), stat(&e)?))
            })
            .collect();
        let policy = GcPolicy::Newest(keep);
        gc_groups(files, policy, GC_METALLIB_MIN_AGE, now, dry, &mut report);
    }
    report
}

/// The runtime archive bases ([archive_base] file names, `<unit>-<key24>`) this process has
/// loaded or is loading ([register_live]); [start_gc] never deletes them.
fn live_set() -> &'static Mutex<std::collections::HashSet<String>> {
    static LIVE: std::sync::OnceLock<Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    LIVE.get_or_init(Default::default)
}

/// W7-Z1: record that this process uses the runtime archive of `base` ([archive_base]), before
/// it opens it: the GC keeps it.
#[doc(hidden)]
pub fn register_live(base: &Path) {
    if let Some(name) = base.file_name().and_then(|n| n.to_str()) {
        live_set().lock().unwrap().insert(name.to_string());
    }
}

/// Whether `unit`'s key `key` was registered by [register_live] in this process.
#[doc(hidden)]
pub fn is_live(unit: &str, key: &str) -> bool {
    live_set()
        .lock()
        .unwrap()
        .contains(&format!("{unit}-{key}"))
}

/// The [ARCHIVE_GC_KEEP_ENV] setting.
///
/// An explicit [ARCHIVE_KEEP_ENV] (a user keeping several binaries warm) raises it: the daily GC
/// never keeps fewer keys than the save-time prune would.
fn gc_keep() -> usize {
    let keep = std::env::var(ARCHIVE_GC_KEEP_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(DEFAULT_GC_KEEP);
    if std::env::var_os(ARCHIVE_KEEP_ENV).is_some() {
        keep.max(keep_archives())
    } else {
        keep
    }
}

/// The build-kernel metallib cache that holds the default archive directory
/// (`<risc0 cache>/metal-archives`): its parent, when `dir` is that default directory.
fn metallib_cache_of(dir: &Path) -> Option<PathBuf> {
    let parent = dir.parent()?;
    (dir.file_name()? == "metal-archives" && parent.file_name()? == "com.risczero.RISC-Zero.risc0")
        .then(|| parent.to_path_buf())
}

/// W7-Z1: the archive GC of a process, started by each process-wide load once it has opened (and
/// so marked used) its runtime archive. The first call decides whether the GC is due: at most
/// once per [GC_INTERVAL] per directory (its `.gc-stamp` file), off with [ARCHIVE_GC_ENV]`=0`,
/// `dry` only logs (and writes no stamp, so it scans at every start). If it is due, the first
/// call runs [gc_archives] on `dir` (and the build cache beside it) on a background thread, and
/// every later process-wide load of this process runs the same rules for its own `unit`
/// ([gc_archives_of]), so each unit the process loads keeps its current key plus `keep - 1`
/// others while units it never loads are only cleared of keys unused for [GC_UNUSED_MAX_AGE].
/// The keys this process registered ([register_live]) are checked when each file is
/// considered, so a unit that starts loading meanwhile is kept too.
fn start_gc(dir: &Path, unit: &str) {
    static MODE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let mut first = false;
    let mode = *MODE.get_or_init(|| {
        first = true;
        let mode = std::env::var(ARCHIVE_GC_ENV).unwrap_or_default();
        let mode = mode.trim().to_ascii_lowercase();
        if matches!(mode.as_str(), "0" | "off" | "false") {
            return None;
        }
        let dry = mode == "dry";
        let due = std::fs::metadata(dir.join(".gc-stamp"))
            .and_then(|m| m.modified())
            .map_or(true, |t| t.elapsed().map_or(true, |a| a >= GC_INTERVAL));
        (due || dry).then_some(dry)
    });
    let Some(dry) = mode else {
        return;
    };
    let dir = dir.to_path_buf();
    let unit = unit.to_string();
    let now = SystemTime::now();
    let _ = std::thread::Builder::new()
        .name("metal-archive-gc".into())
        .spawn(move || {
            let t = Instant::now();
            let r = if first {
                if !dry
                    && (std::fs::create_dir_all(&dir).is_err()
                        || std::fs::write(dir.join(".gc-stamp"), b"").is_err())
                {
                    return;
                }
                let cache = metallib_cache_of(&dir);
                gc_archives(&dir, cache.as_deref(), now, gc_keep(), dry, &is_live)
            } else {
                gc_archives_of(&dir, None, now, gc_keep(), dry, &is_live, Some(&unit))
            };
            for (p, size) in &r.deleted {
                tracing::debug!(
                    "Metal archive GC: {} {} ({size} B)",
                    if dry { "would delete" } else { "deleted" },
                    p.display()
                );
            }
            let msg = format!(
                "Metal archive GC{}{}: {} files, {:.1} MB {}; {} kept, {} recent ({:.2} s)",
                if dry { " (dry run)" } else { "" },
                if first {
                    String::new()
                } else {
                    format!(" of {unit}")
                },
                r.deleted.len(),
                r.bytes as f64 / 1e6,
                if dry { "to delete" } else { "deleted" },
                r.kept,
                r.recent,
                t.elapsed().as_secs_f64()
            );
            if first || !r.deleted.is_empty() {
                tracing::info!("{msg}");
            } else {
                tracing::debug!("{msg}");
            }
        });
}
