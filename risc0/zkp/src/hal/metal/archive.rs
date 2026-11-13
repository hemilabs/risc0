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
//! for one GPU architecture each, named
//! `<unit>-<chunked_digest(metallib)[..24]>-<MTLDevice.architecture.name>-<checksum>.binarchive`.
//! The one matching this metallib and this GPU is checked against its checksum, loaded, and tried
//! first for every PSO (`FailOnBinaryArchiveMiss`); a miss continues as above (the runtime
//! archive, then a compile). So a first install on a GPU the shipped set covers compiles
//! nothing, and any other GPU, OS or metallib just misses (a wrong archive is never fatal: Metal
//! reports a miss, Z16 note). The directory is never written. Unset: no change.
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
    pub fn new_in_threads(
        dir: Option<PathBuf>,
        device: &DeviceRef,
        unit: &str,
        bytes: &[u8],
        names: &[&str],
        threads: Option<usize>,
    ) -> Result<Self> {
        Self::new_in_aot(dir, aot_dir(), device, unit, bytes, names, threads)
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
        let opened = SystemTime::now();
        let archive = base
            .as_deref()
            .and_then(|base| open_archive(device, base, &mut report));
        let aot = aot_dir.and_then(|d| open_aot_archive(&d, device, unit, bytes, &mut report));

        let add_nanos = AtomicU64::new(0);
        let built = build_all(
            device,
            &library,
            names,
            Archives {
                aot: aot.as_deref(),
                runtime: archive.as_deref(),
            },
            &add_nanos,
            threads,
        )?;
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
            "{unit}: {} PSOs in {:.2} s (archive {}: {} hits, {} misses, loaded {}, adds {:.2} s, saved {} in {:.2} s)",
            names.len(),
            report.elapsed.as_secs_f64(),
            report
                .path
                .as_deref()
                .map_or("off".into(), |p| p.display().to_string()),
            report.hits,
            report.misses,
            report.loaded,
            report.add_time.as_secs_f64(),
            report.saved,
            report.save_time.as_secs_f64(),
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

/// Z16: `<unit>-<chunked_digest(bytes)[..24]>-<arch>`, the stem of the offline archive of a
/// metallib for one GPU architecture (`<stem>-<checksum>.binarchive`, as [archive_file]).
#[doc(hidden)]
pub fn aot_stem(unit: &str, bytes: &[u8], arch: &str) -> String {
    format!("{unit}-{}-{arch}", &chunked_digest(bytes)[..24])
}

/// Z16: load the offline archive of `unit`/`bytes` for this GPU from `dir`, if there is one whose
/// content matches the checksum in its name (a damaged archive can abort the process inside
/// Metal, K8; the module docs). Never fatal and never writes `dir`: a missing, mismatched or
/// unloadable file is reported (debug/warn) and skipped.
fn open_aot_archive(
    dir: &Path,
    device: &DeviceRef,
    unit: &str,
    bytes: &[u8],
    report: &mut ArchiveReport,
) -> Option<BinaryArchive> {
    let arch = gpu_architecture(device)?;
    let stem = aot_stem(unit, bytes, &arch);
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(sum) = name.to_str().and_then(|n| checksum_in_name(&stem, n)) else {
            continue;
        };
        let path = entry.path();
        let ok = std::fs::read(&path).is_ok_and(|c| archive_checksum(&c) == sum);
        if !ok {
            let msg = format!(
                "offline archive {}: checksum mismatch; skipped",
                path.display()
            );
            tracing::warn!("{unit}: Metal binary archive: {msg}");
            report.errors.push(msg);
            continue;
        }
        let desc = BinaryArchiveDescriptor::new();
        desc.set_url(&file_url(&path));
        match device.new_binary_archive_with_descriptor(&desc) {
            Ok(a) => {
                report.aot_path = Some(path);
                return Some(a);
            }
            Err(e) => {
                let msg = format!("offline archive {}: {e}; skipped", path.display());
                tracing::warn!("{unit}: Metal binary archive: {msg}");
                report.errors.push(msg);
            }
        }
    }
    tracing::debug!(
        "{unit}: no offline archive {stem}-*.{ARCHIVE_EXT} in {}",
        dir.display()
    );
    None
}

/// The OS build (`ProductBuildVersion`, e.g. `25A354`): a new build can bring a new GPU compiler,
/// whose binaries the old archive does not hold. None when neither the plist nor `sysctl` gives
/// it: the archive is then not used at all ([archive_base]), because a key without the OS build
/// would survive an OS update (K8).
#[doc(hidden)]
pub fn os_build() -> Option<String> {
    let plist = std::fs::read_to_string("/System/Library/CoreServices/SystemVersion.plist")
        .unwrap_or_default();
    let from_plist = plist
        .split("<key>ProductBuildVersion</key>")
        .nth(1)
        .and_then(|rest| rest.split("<string>").nth(1))
        .and_then(|rest| rest.split("</string>").next())
        .map(|s| s.trim().to_string());
    from_plist.filter(|s| !s.is_empty()).or_else(|| {
        std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "kern.osversion"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| !s.is_empty())
    })
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

/// The archives a PSO may come from: the read-only offline one (Z16, tried first) and the
/// runtime one (hit, else compiled into it).
#[derive(Clone, Copy)]
struct Archives<'a> {
    aot: Option<&'a BinaryArchiveRef>,
    runtime: Option<&'a BinaryArchiveRef>,
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
    if let Some(aot) = archives.aot {
        desc.set_binary_archives(&[aot]);
        if let Ok(pso) =
            new_pso_with_options(device, &desc, MTLPipelineOption::FailOnBinaryArchiveMiss)
        {
            return Ok((pso, Source::AotHit));
        }
        desc.set_binary_archives(&[]);
    }
    let archive = archives.runtime;
    let mut archive_err = None;
    if let Some(archive) = archive {
        desc.set_binary_archives(&[archive]);
        let from_archive =
            || new_pso_with_options(device, &desc, MTLPipelineOption::FailOnBinaryArchiveMiss);
        if let Ok(pso) = from_archive() {
            return Ok((pso, Source::Hit));
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
pub fn specialized_pipeline(
    dir: Option<PathBuf>,
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
    let opened = SystemTime::now();
    let archive = base
        .as_deref()
        .and_then(|base| open_archive(device, base, &mut report));
    let add_nanos = AtomicU64::new(0);
    let archives = Archives {
        aot: None,
        runtime: archive.as_deref(),
    };
    let (pso, source) = metal::objc::rc::autoreleasepool(|| {
        build_function(device, &function, &key_name, archives, &add_nanos)
    })?;
    report.add_time = Duration::from_nanos(add_nanos.load(Ordering::Relaxed));
    let mut added = false;
    match source {
        Source::Hit | Source::AotHit => report.hits += 1,
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
