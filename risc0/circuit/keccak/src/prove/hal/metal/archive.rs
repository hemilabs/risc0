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

//! GPU-binary caching of the keccak pipelines with `MTLBinaryArchive` (PLAN 4.4 item K5, 9.1).
//!
//! Creating the keccak PSOs compiles the metallib's AIR to GPU code: 40-190 s cold for the 58
//! eval_check stages and 20-35 s for the 12 witgen arms on an M5 Max (K3, K4, K6 notes), more on
//! the 8-10-core base chips of the minimum target. The OS shader cache usually makes a second
//! process instant, but it was seen to miss (cache shared with every Metal app, evictions,
//! relinked executables), so the worker cannot rely on it to keep cold PSO creation out of a job
//! deadline. This module keeps the compiled GPU binaries in a binary archive file of our own:
//!
//! - **Location**: `RISC0_METAL_ARCHIVE_DIR`, else the risc0 cache directory
//!   (`~/Library/Caches/com.risczero.RISC-Zero.risc0`, the directory `risc0-build-kernel` caches
//!   compiler outputs in) `/metal-archives`.
//! - **Key**: `<unit>-<sha256(digest(metallib) ‖ device name ‖ OS build ‖ kernel names)[..24]>`,
//!   `digest` = [chunked_digest] (K7: sha256 of the per-MiB sha256s, hashed on all cores).
//!   A new metallib, another GPU or an OS update (which can change the GPU compiler) therefore
//!   never reads an old archive. When a new one is written, the unit keeps its
//!   `RISC0_METAL_ARCHIVE_KEEP` (default 4) most recently used other keys and deletes the rest,
//!   and any key unused for 30 days ([prune_other_keys]; K7, K8 finding 4).
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
//! Nothing here is keccak-specific: rv32im and recursion (C11, M1) can move it into the zkp
//! Metal HAL and use it with their own unit names.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Mutex,
    },
    time::{Duration, Instant, SystemTime},
};

use anyhow::{anyhow, bail, Result};
use metal::{
    foreign_types::ForeignType,
    objc::{msg_send, runtime::Object, sel, sel_impl},
    BinaryArchive, BinaryArchiveDescriptor, BinaryArchiveRef, ComputePipelineDescriptor,
    ComputePipelineDescriptorRef, ComputePipelineState, DeviceRef, Library, MTLPipelineOption, URL,
};
use risc0_zkp::core::hash::sha::{cpu::Impl as Sha256Impl, Sha256 as _};

/// Disables the binary archive when `0`.
pub(crate) const ARCHIVE_ENV: &str = "RISC0_METAL_ARCHIVE";
/// Overrides the archive directory.
pub(crate) const ARCHIVE_DIR_ENV: &str = "RISC0_METAL_ARCHIVE_DIR";
const ARCHIVE_EXT: &str = "binarchive";

/// What happened to the archive while the PSOs were built.
#[derive(Clone, Debug, Default)]
pub struct ArchiveReport {
    /// The archive file (None: disabled or no cache directory).
    pub path: Option<PathBuf>,
    /// An existing archive file was loaded.
    pub loaded: bool,
    /// PSOs loaded from the archive (no GPU compile).
    pub hits: usize,
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
    #[allow(dead_code)]
    library: Library,
    pipelines: HashMap<String, ComputePipelineState>,
    pub report: ArchiveReport,
}

impl ArchivedKernels {
    /// The PSO for `name`; panics (a programming error, plan 2.9) if it was not requested.
    pub fn pipeline(&self, name: &str) -> &ComputePipelineState {
        self.get(name)
            .unwrap_or_else(|| panic!("Metal kernel {name} was not loaded"))
    }

    pub fn get(&self, name: &str) -> Option<&ComputePipelineState> {
        self.pipelines.get(name)
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
        let t0 = Instant::now();
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

        let add_nanos = AtomicU64::new(0);
        let built = build_all(device, &library, names, archive.as_deref(), &add_nanos)?;
        let mut pipelines = HashMap::with_capacity(names.len());
        let mut added = 0;
        for (name, pso, source) in built {
            match source {
                Source::Hit => report.hits += 1,
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
pub(crate) fn archive_dir() -> Option<PathBuf> {
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

/// The OS build (`ProductBuildVersion`, e.g. `25A354`): a new build can bring a new GPU compiler,
/// whose binaries the old archive does not hold. None when neither the plist nor `sysctl` gives
/// it: the archive is then not used at all ([archive_base]), because a key without the OS build
/// would survive an OS update (K8).
pub(crate) fn os_build() -> Option<String> {
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
const DIGEST_CHUNK: usize = 1 << 20;

/// Environment switch back to K8's single-stream sha256 for the key and checksum (K7 revert knob):
/// `RISC0_METAL_ARCHIVE_HASH=sha256`. Archives written under one setting have another file name
/// (key) and checksum than under the other, so switching costs one cold build.
pub(crate) const ARCHIVE_HASH_ENV: &str = "RISC0_METAL_ARCHIVE_HASH";

/// The hex digest the archive key and checksum are built from (K7): sha256 over the sha256s of
/// the 1 MiB chunks of `bytes`, the chunks hashed on all cores (K8 finding 5: hashing the
/// 16 MB metallib and the ~50 MB archive in one stream took ~130 ms of every warm start).
/// Any changed byte changes its chunk's digest and so the result, like a plain sha256 (a Merkle
/// list of one level). With [ARCHIVE_HASH_ENV]=sha256: plain sha256 of `bytes`.
pub(crate) fn chunked_digest(bytes: &[u8]) -> String {
    if std::env::var(ARCHIVE_HASH_ENV).is_ok_and(|v| v == "sha256") {
        return Sha256Impl::hash_bytes(bytes).to_string();
    }
    parallel_digest(bytes)
}

/// [chunked_digest] without the [ARCHIVE_HASH_ENV] switch: always the parallel chunked form.
fn parallel_digest(bytes: &[u8]) -> String {
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
pub(crate) fn archive_key(device_name: &str, os: &str, bytes: &[u8], names: &[&str]) -> String {
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
pub(crate) fn archive_base(
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
pub(crate) fn archive_base_for(
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
pub(crate) fn archive_checksum(content: &[u8]) -> String {
    chunked_digest(content)[..CHECKSUM_LEN].to_string()
}

/// The file name of an archive with base `base` ([archive_base]) and content `content`.
pub(crate) fn archive_file(base: &Path, content: &[u8]) -> PathBuf {
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
fn file_url(path: &Path) -> URL {
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
fn open_archive_with(
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
fn new_pso_with_options(
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
    /// Compiled into the archive (`addComputePipelineFunctions`), then loaded from it.
    Added,
    /// Built without the archive: there is none (`None`), or the archive failed (the error).
    Plain(Option<String>),
}

type Built<'a> = (&'a str, ComputePipelineState, Source);

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
    archive: Option<&BinaryArchiveRef>,
    add_nanos: &AtomicU64,
) -> Result<Vec<Built<'a>>> {
    let build_one = |name: &'a str| -> Result<Built<'a>> {
        let function = library
            .get_function(name, None)
            .map_err(|e| anyhow!("metallib function {name}: {e}"))?;
        let desc = ComputePipelineDescriptor::new();
        desc.set_compute_function(Some(&function));
        let mut archive_err = None;
        if let Some(archive) = archive {
            desc.set_binary_archives(&[archive]);
            let from_archive =
                || new_pso_with_options(device, &desc, MTLPipelineOption::FailOnBinaryArchiveMiss);
            if let Ok(pso) = from_archive() {
                return Ok((name, pso, Source::Hit));
            }
            let t = Instant::now();
            let added = archive.add_compute_pipeline_functions_with_descriptor(&desc);
            add_nanos.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            match added {
                Ok(_) => match from_archive() {
                    Ok(pso) => return Ok((name, pso, Source::Added)),
                    Err(e) => archive_err = Some(format!("not in the archive after adding: {e}")),
                },
                Err(e) => archive_err = Some(format!("adding to the archive: {e}")),
            }
            desc.set_binary_archives(&[]);
        }
        let pso = device
            .new_compute_pipeline_state(&desc)
            .map_err(|e| anyhow!("compute pipeline for {name}: {e}"))?;
        Ok((name, pso, Source::Plain(archive_err)))
    };

    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
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
/// [archive_file]), then delete the unit's other (stale) archives in the directory. A file of the
/// same key written at or after `opened` (when this process looked for one) is another process's
/// concurrent save and is kept: deleting it would make the processes overwrite each other's
/// archives in turn. The next [verified_archive_file] keeps one of the two. Returns the file
/// written.
fn save_archive(
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
pub(crate) const ARCHIVE_KEEP_ENV: &str = "RISC0_METAL_ARCHIVE_KEEP";

/// Default of [ARCHIVE_KEEP_ENV]. K8 finding 4: deleting every other key made two binaries with
/// different metallibs on one machine (risc0 versions, worktrees, a worker and a test build)
/// evict each other, so each switch paid a cold build (~25 s on an M5 Max, minutes on base
/// chips). Four keys of ~35-60 MB each bound the directory to a few hundred MB per unit.
pub(crate) const DEFAULT_ARCHIVE_KEEP: usize = 4;

/// Archives of other keys unused for this long are deleted regardless of the count (an OS
/// update or a new risc0 version leaves the old ones behind for good).
pub(crate) const ARCHIVE_MAX_AGE: Duration = Duration::from_secs(30 * 24 * 3600);

fn keep_archives() -> usize {
    std::env::var(ARCHIVE_KEEP_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(DEFAULT_ARCHIVE_KEEP)
}

/// Deletes the unit's archives of keys other than `stem`'s beyond the `keep - 1` most recently
/// used ones (modification time: written or loaded, see [open_archive_with]), and any of them
/// unused for [ARCHIVE_MAX_AGE] (relative to `now`). Never touches `stem`'s own files.
pub(crate) fn prune_other_keys(dir: &Path, unit: &str, stem: &str, keep: usize, now: SystemTime) {
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

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use metal::Device;
    use risc0_circuit_keccak_sys::METAL_LIB;

    use super::{archive_key, ArchivedKernels};

    /// A fresh directory for one test's archive (scratch, removed at the start).
    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("risc0-k5-archive-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// A small, quick set of keccak kernels (the scatter kernel, the last eval stage, two arms).
    fn small_set() -> Vec<&'static str> {
        let mut names = vec!["scatter_preflight"];
        names.extend(
            risc0_circuit_keccak_sys::metal_eval::EVAL_STAGE_NAMES
                .last()
                .copied(),
        );
        names.extend(risc0_circuit_keccak_sys::metal_witgen::WITGEN_ARM_KERNELS[..2].iter());
        names
    }

    /// Every archive file in `dir` matches the checksum in its name.
    fn assert_archives_intact(dir: &std::path::Path) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "binarchive") {
                let content = std::fs::read(&path).unwrap();
                let sum = super::archive_checksum(&content);
                assert!(
                    path.to_string_lossy()
                        .ends_with(&format!("-{sum}.binarchive")),
                    "{}: content does not match the name",
                    path.display()
                );
            }
        }
    }

    fn device() -> Device {
        Device::system_default().expect("no Metal device")
    }

    /// First process: every PSO misses and the archive is written; second: the file loads and
    /// every PSO is a hit (`FailOnBinaryArchiveMiss`), nothing is written.
    #[test]
    fn archive_roundtrip() {
        let _gpu = super::super::tests::gpu_test_lock();
        let device = device();
        let dir = temp_dir("roundtrip");
        let names = small_set();
        let first =
            ArchivedKernels::new_in(Some(dir.clone()), &device, "keccak", METAL_LIB, &names)
                .unwrap();
        let r = &first.report;
        assert!(r.errors.is_empty(), "{r:?}");
        assert!(!r.loaded);
        assert_eq!((r.hits, r.misses), (0, names.len()), "{r:?}");
        assert!(r.saved);
        let path = r.path.clone().unwrap();
        assert!(path.starts_with(&dir));
        // The file name carries the checksum of its content.
        let content = std::fs::read(&path).unwrap();
        assert!(!content.is_empty());
        assert_eq!(
            path,
            super::archive_file(
                &super::archive_base(&dir, &device, "keccak", METAL_LIB, &names).unwrap(),
                &content
            )
        );

        let second =
            ArchivedKernels::new_in(Some(dir.clone()), &device, "keccak", METAL_LIB, &names)
                .unwrap();
        let r = &second.report;
        assert!(r.errors.is_empty(), "{r:?}");
        assert!(r.loaded);
        assert_eq!(r.path.as_ref(), Some(&path));
        assert_eq!((r.hits, r.misses), (names.len(), 0), "{r:?}");
        assert!(!r.saved);
        for name in &names {
            let (a, b) = (first.pipeline(name), second.pipeline(name));
            assert_eq!(
                a.max_total_threads_per_threadgroup(),
                b.max_total_threads_per_threadgroup(),
                "{name}"
            );
        }
        // Another kernel set has another key: a new archive; the old one (same unit, another
        // key) stays, within the keep count (K7: no eviction ping-pong between two binaries).
        let third =
            ArchivedKernels::new_in(Some(dir.clone()), &device, "keccak", METAL_LIB, &names[..1])
                .unwrap();
        assert_ne!(third.report.path.as_ref(), Some(&path));
        assert!(third.report.saved);
        if super::keep_archives() > 1 {
            assert!(path.exists(), "another key's archive was evicted");
        } else {
            assert!(!path.exists(), "stale archive kept");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The content of a valid archive holding only `names[..1]` (another key; only the content
    /// is used, under the name a file of `base` would have).
    fn partial_archive(device: &Device, names: &[&str], tag: &str) -> Vec<u8> {
        let other = temp_dir(tag);
        let k = ArchivedKernels::new_in(
            Some(other.clone()),
            device,
            "keccak",
            METAL_LIB,
            &names[..1],
        )
        .unwrap();
        let content = std::fs::read(k.report.path.as_ref().unwrap()).unwrap();
        let _ = std::fs::remove_dir_all(&other);
        content
    }

    /// Another process replaces the archive between verification and Metal's open (seen in the
    /// K8 round-2 verification: "Invalid URL; deleting it", failing warmup_builds_every_pso).
    /// Not an error: the newer file is loaded, or, if none is left, an empty archive is used.
    /// Before the fix both cases reported an error and the first one did not load.
    #[test]
    fn archive_replaced_before_load() {
        let _gpu = super::super::tests::gpu_test_lock();
        let device = device();
        let dir = temp_dir("replaced");
        let names = small_set();
        let base = super::archive_base(&dir, &device, "keccak", METAL_LIB, &names).unwrap();
        let first =
            ArchivedKernels::new_in(Some(dir.clone()), &device, "keccak", METAL_LIB, &names)
                .unwrap();
        let verified = first.report.path.clone().unwrap();
        let newer = partial_archive(&device, &names, "replaced-partial");
        let newer_path = super::archive_file(&base, &newer);
        assert_ne!(newer_path, verified);

        // (1) Replaced by another process's save: the newer file loads.
        let mut calls = 0;
        let mut report = super::ArchiveReport::default();
        let archive = super::open_archive_with(&device, &base, &mut report, &mut |path| {
            calls += 1;
            if calls == 1 {
                assert_eq!(path, verified.as_path());
                std::fs::write(&newer_path, &newer).unwrap();
                std::fs::remove_file(path).unwrap();
            }
        });
        assert!(archive.is_some());
        assert!(report.errors.is_empty(), "{report:?}");
        assert!(report.loaded, "{report:?}");
        assert_eq!(report.path.as_ref(), Some(&newer_path));
        assert_eq!(calls, 2);
        assert!(newer_path.exists());

        // (2) Deleted and nothing newer: an empty archive, still no error.
        let mut report = super::ArchiveReport::default();
        let archive = super::open_archive_with(&device, &base, &mut report, &mut |path| {
            std::fs::remove_file(path).unwrap();
        });
        assert!(archive.is_some());
        assert!(report.errors.is_empty(), "{report:?}");
        assert!(!report.loaded);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A save keeps a same-key archive another process saved after this one looked for one
    /// (before the fix it deleted it, so concurrent processes deleted each other's archives),
    /// and still deletes one that is older than that.
    #[test]
    fn save_keeps_concurrent_same_key_archive() {
        let _gpu = super::super::tests::gpu_test_lock();
        let device = device();
        let dir = temp_dir("concurrent-save");
        let names = small_set();
        let base = super::archive_base(&dir, &device, "keccak", METAL_LIB, &names).unwrap();
        let other = partial_archive(&device, &names, "concurrent-save-partial");
        let other_path = super::archive_file(&base, &other);
        // An archive with one added function to save (an empty one has "Nothing to
        // serialize"), and not the function of `other`, so the two files differ. A fresh one per
        // save: Metal does not serialize one archive twice after its file was renamed.
        let library = device.new_library_with_data(METAL_LIB).unwrap();
        let new_archive = || {
            let archive = device
                .new_binary_archive_with_descriptor(&metal::BinaryArchiveDescriptor::new())
                .unwrap();
            let desc = metal::ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&library.get_function(names[1], None).unwrap()));
            archive
                .add_compute_pipeline_functions_with_descriptor(&desc)
                .unwrap();
            archive
        };

        std::fs::create_dir_all(&dir).unwrap();
        let opened = std::time::SystemTime::now() - std::time::Duration::from_secs(1);
        std::fs::write(&other_path, &other).unwrap();
        let saved = super::save_archive(&new_archive(), &base, "keccak", opened).unwrap();
        assert_ne!(saved, other_path);
        assert!(saved.exists());
        assert!(
            other_path.exists(),
            "a concurrent save of the same key was deleted"
        );

        // Written before this process looked: stale, deleted.
        let opened = std::time::SystemTime::now() + std::time::Duration::from_secs(1);
        let saved = super::save_archive(&new_archive(), &base, "keccak", opened).unwrap();
        assert!(saved.exists());
        assert!(!other_path.exists(), "a stale archive was kept");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A corrupt archive file is reported, deleted and rewritten; the PSOs are built anyway:
    /// (1) garbage whose name carries its checksum (so Metal itself rejects it), (2) garbage
    /// under another checksum (rejected before Metal sees it).
    #[test]
    fn corrupt_archive_falls_back() {
        let _gpu = super::super::tests::gpu_test_lock();
        let device = device();
        let dir = temp_dir("corrupt");
        let names = small_set();
        let base = super::archive_base(&dir, &device, "keccak", METAL_LIB, &names).unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        let garbage = b"not a binary archive";
        let wrong_sum = super::archive_file(&base, b"other content");
        for (path, what) in [
            (super::archive_file(&base, garbage), "loading"),
            (wrong_sum, "does not match its checksum"),
        ] {
            std::fs::write(&path, garbage).unwrap();
            let k =
                ArchivedKernels::new_in(Some(dir.clone()), &device, "keccak", METAL_LIB, &names)
                    .unwrap();
            let r = &k.report;
            assert!(!r.loaded);
            assert_eq!(r.errors.len(), 1, "{r:?}");
            assert!(r.errors[0].contains(what), "{r:?}");
            assert_eq!((r.hits, r.misses), (0, names.len()));
            assert!(r.saved);
            assert_archives_intact(&dir);
            assert_ne!(r.path.as_ref(), Some(&path));
            for name in &names {
                let _ = k.pipeline(name);
            }
            let again =
                ArchivedKernels::new_in(Some(dir.clone()), &device, "keccak", METAL_LIB, &names)
                    .unwrap();
            assert_eq!((again.report.hits, again.report.misses), (names.len(), 0));
            assert!(again.report.errors.is_empty(), "{:?}", again.report);
            std::fs::remove_file(again.report.path.as_ref().unwrap()).unwrap();
        }

        // An unwritable directory (a file in its place): the PSOs are built, the save fails.
        let blocked = dir.join("blocked");
        std::fs::write(&blocked, b"").unwrap();
        let k =
            ArchivedKernels::new_in(Some(blocked), &device, "keccak", METAL_LIB, &names).unwrap();
        assert!(!k.report.saved);
        assert_eq!(k.report.errors.len(), 1, "{:?}", k.report);
        for name in &names {
            let _ = k.pipeline(name);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A real archive damaged on disk (truncated, bytes flipped inside, emptied) is rejected by
    /// its checksum, deleted and rebuilt; the PSOs are always built. Before K8, some of these
    /// loaded ("flipped middle", "last byte missing": hits from a damaged file) and
    /// `archive_bitflip_sweep`'s flips aborted the process inside Metal.
    #[test]
    fn damaged_archive_falls_back() {
        let _gpu = super::super::tests::gpu_test_lock();
        let device = device();
        let dir = temp_dir("damaged");
        let names = small_set();
        let good = ArchivedKernels::new_in(Some(dir.clone()), &device, "keccak", METAL_LIB, &names)
            .unwrap();
        let path = good.report.path.clone().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 4096, "archive of {} bytes", bytes.len());
        let flip = |from: usize, to: usize| {
            let mut b = bytes.clone();
            for x in &mut b[from..to] {
                *x ^= 0x5a;
            }
            b
        };
        let n = bytes.len();
        let variants: Vec<(&str, Vec<u8>)> = vec![
            ("empty", Vec::new()),
            ("header only", bytes[..64].to_vec()),
            ("truncated to half", bytes[..n / 2].to_vec()),
            ("last byte missing", bytes[..n - 1].to_vec()),
            ("flipped header", flip(0, 64)),
            ("flipped middle", flip(n / 2, n / 2 + 4096)),
            ("flipped tail", flip(n - 4096, n)),
        ];
        for (what, damaged) in variants {
            // Only the damaged file, under the intact archive's name.
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(&path, &damaged).unwrap();
            let k =
                ArchivedKernels::new_in(Some(dir.clone()), &device, "keccak", METAL_LIB, &names)
                    .unwrap_or_else(|e| panic!("{what}: {e:#}"));
            let r = &k.report;
            println!(
                "{what}: loaded {} hits {} misses {} saved {} errors {:?}",
                r.loaded, r.hits, r.misses, r.saved, r.errors
            );
            // Never handed to Metal: it does not match the checksum in its name.
            assert!(!r.loaded, "{what}: {r:?}");
            assert!(r.errors[0].contains("checksum"), "{what}: {r:?}");
            // (The rebuilt archive may serialize to the very bytes, so the same name, as the
            // intact one: what is left must match its name.)
            assert_archives_intact(&dir);
            assert_eq!((r.hits, r.misses), (0, names.len()), "{what}: {r:?}");
            for name in &names {
                let pso = k.pipeline(name);
                assert!(
                    pso.max_total_threads_per_threadgroup() >= 32,
                    "{what}: {name}"
                );
            }
            // Whatever happened, the file on disk is usable again: all hits next time.
            let again =
                ArchivedKernels::new_in(Some(dir.clone()), &device, "keccak", METAL_LIB, &names)
                    .unwrap();
            assert_eq!(
                (again.report.hits, again.report.misses),
                (names.len(), 0),
                "{what}: after the repair: {:?}",
                again.report
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Run `scatter_preflight` (built by `k`) on a fixed input with raw Metal calls; returns the
    /// written cells, or the command buffer's error.
    fn run_scatter(device: &metal::DeviceRef, k: &ArchivedKernels) -> Result<Vec<u32>, String> {
        use risc0_circuit_keccak_sys::ScatterInfo;
        let (rows, cols) = (64u32, 16u32);
        let infos: Vec<ScatterInfo> = (0..rows)
            .map(|r| ScatterInfo {
                offset: r,
                row: r,
                col: (r % 4) as u16,
                count: 12,
                bits: [1u32, 2, 4, 8][r as usize % 4],
            })
            .collect();
        let from: Vec<u32> = (0..rows + 32)
            .map(|i| 0x9e37_79b9u32.wrapping_mul(i + 1))
            .collect();
        let opts = metal::MTLResourceOptions::StorageModeShared;
        let new_buf = |bytes: &[u8]| {
            device.new_buffer_with_data(bytes.as_ptr().cast(), bytes.len() as u64, opts)
        };
        let into = new_buf(&vec![0xffu8; (rows * cols * 4) as usize]);
        let infos_buf =
            new_buf(unsafe { std::slice::from_raw_parts(infos.as_ptr().cast(), infos.len() * 16) });
        let from_buf = new_buf(bytemuck::cast_slice(&from));
        let fault = new_buf(&[0u8; 4]);
        let params = [rows, cols, infos.len() as u32, from.len() as u32];
        let queue = device.new_command_queue();
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let pso = k.pipeline("scatter_preflight");
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(&into), 0);
        enc.set_buffer(1, Some(&infos_buf), 0);
        enc.set_buffer(2, Some(&from_buf), 0);
        enc.set_buffer(3, Some(&fault), 0);
        enc.set_bytes(4, 16, params.as_ptr().cast());
        enc.dispatch_threads(
            metal::MTLSize::new(infos.len() as u64, 1, 1),
            metal::MTLSize::new(64, 1, 1),
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        if cmd.status() != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("command buffer status {:?}", cmd.status()));
        }
        let mut out = vec![0u32; (rows * cols) as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                into.contents().cast::<u32>(),
                out.as_mut_ptr(),
                out.len(),
            );
            out.push(*fault.contents().cast::<u32>());
        }
        Ok(out)
    }

    /// Flip 16 bytes at every `step`-th offset of a `scatter_preflight`-only archive (about
    /// 20 KB); for each damaged file, build the PSO through the archive dir and, whenever the
    /// archive served a hit, run the kernel and compare with the intact PSO's output. With
    /// `bypass`, the damaged file is renamed to carry its own checksum, so Metal itself gets it
    /// (this is what aborts the process: see the module doc). Returns
    /// (rejected, loaded-but-recompiled, hits with the right output, bad hits).
    fn bitflip_sweep(step: usize, from: usize, bypass: bool) -> (usize, usize, usize, Vec<String>) {
        let device = device();
        let dir = temp_dir(if bypass { "bitflip-bypass" } else { "bitflip" });
        let names = ["scatter_preflight"];
        let good = ArchivedKernels::new_in(Some(dir.clone()), &device, "keccak", METAL_LIB, &names)
            .unwrap();
        let expected = run_scatter(&device, &good).unwrap();
        let path = good.report.path.clone().unwrap();
        let base = super::archive_base(&dir, &device, "keccak", METAL_LIB, &names).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let window = 16.min(step);
        let trace = std::env::var("K8_FLIP_TRACE").is_ok();
        let (mut rejected, mut recompiled, mut hits_ok, mut hits_bad) = (0, 0, 0, Vec::new());
        let mut at = from;
        while at < bytes.len() {
            let mut b = bytes.clone();
            for x in &mut b[at..(at + window).min(bytes.len())] {
                *x ^= 0xa5;
            }
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let target = if bypass {
                super::archive_file(&base, &b)
            } else {
                path.clone()
            };
            std::fs::write(&target, &b).unwrap();
            if trace {
                eprintln!("flip at {at}");
            }
            let k =
                ArchivedKernels::new_in(Some(dir.clone()), &device, "keccak", METAL_LIB, &names)
                    .unwrap();
            if !k.report.loaded {
                rejected += 1;
            } else if k.report.hits == 0 {
                recompiled += 1;
            } else {
                if trace {
                    eprintln!("  hit at {at}: running the kernel");
                }
                match run_scatter(&device, &k) {
                    Ok(out) if out == expected => hits_ok += 1,
                    Ok(_) => hits_bad.push(format!("{at}: wrong output")),
                    Err(e) => hits_bad.push(format!("{at}: {e}")),
                }
            }
            at += step;
        }
        println!(
            "archive of {} bytes, {window}-byte flips every {step} from {from} (bypass {bypass}): \
             file rejected {rejected}, loaded but kernel recompiled {recompiled}, hit with the \
             right output {hits_ok}, hit with a wrong output or GPU error {}: {:?}",
            bytes.len(),
            hits_bad.len(),
            &hits_bad[..hits_bad.len().min(10)]
        );
        let _ = std::fs::remove_dir_all(&dir);
        (rejected, recompiled, hits_ok, hits_bad)
    }

    /// Damage anywhere in an archive file is caught by its checksum: every flipped file is
    /// rejected, and the process survives (before K8 the flip at byte 10240 of this archive
    /// aborted it inside Metal: `std::length_error`, uncaught).
    #[test]
    fn archive_bitflip_sweep() {
        let _gpu = super::super::tests::gpu_test_lock();
        let (rejected, recompiled, hits_ok, bad) = bitflip_sweep(512, 0, false);
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!((recompiled, hits_ok), (0, 0));
        assert!(rejected > 0);
    }

    /// Probe of Metal's own handling of damaged archives (`--ignored`; aborts the process on
    /// macOS 27.2 build 26B131): the damaged file is renamed to match its checksum.
    /// `K8_FLIP_STEP` (default 512), `K8_FLIP_FROM` (default 0), `K8_FLIP_TRACE=1`.
    #[test]
    #[ignore]
    fn archive_bitflip_probe() {
        let _gpu = super::super::tests::gpu_test_lock();
        let env = |k: &str, d: usize| std::env::var(k).map_or(d, |s| s.parse().unwrap());
        let (_, _, _, bad) = bitflip_sweep(env("K8_FLIP_STEP", 512), env("K8_FLIP_FROM", 0), true);
        assert!(bad.is_empty(), "{bad:?}");
    }

    /// No directory: no archive, plain PSOs.
    #[test]
    fn archive_disabled() {
        let _gpu = super::super::tests::gpu_test_lock();
        let device = device();
        let names = small_set();
        let k = ArchivedKernels::new_in(None, &device, "keccak", METAL_LIB, &names).unwrap();
        let r = &k.report;
        assert!(r.path.is_none() && r.errors.is_empty() && !r.saved);
        assert_eq!((r.hits, r.misses), (0, names.len()));
        // A missing kernel still fails (not an archive matter).
        let err = ArchivedKernels::new_in(None, &device, "keccak", METAL_LIB, &["no_such_kernel"])
            .err()
            .unwrap();
        assert!(err.to_string().contains("no_such_kernel"), "{err}");
    }

    /// The key covers the metallib, the device, the OS build and the kernel names.
    /// Other keys' archives: the `keep - 1` most recently used stay, older ones and any unused
    /// for 30 days go; the stem's own files are never touched.
    #[test]
    fn prune_keeps_recent_keys() {
        use std::time::{Duration, SystemTime};
        let dir = temp_dir("prune");
        std::fs::create_dir_all(&dir).unwrap();
        let now = SystemTime::now();
        let file = |name: &str, age_s: u64| {
            let p = dir.join(name);
            std::fs::write(&p, b"x").unwrap();
            std::fs::File::options()
                .write(true)
                .open(&p)
                .unwrap()
                .set_modified(now - Duration::from_secs(age_s))
                .unwrap();
            p
        };
        let own = file("keccak-k0-0000000000000000.binarchive", 90 * 86400);
        let a = file("keccak-k1-0000000000000000.binarchive", 10);
        let b = file("keccak-k2-0000000000000000.binarchive", 20);
        let c = file("keccak-k3-0000000000000000.binarchive", 30);
        let d = file("keccak-k4-0000000000000000.binarchive", 40);
        let old = file("keccak-k5-0000000000000000.binarchive", 31 * 86400);
        let foreign = file("rv32im-k6-0000000000000000.binarchive", 50 * 86400);
        super::prune_other_keys(&dir, "keccak", "keccak-k0", 4, now);
        assert!(own.exists() && a.exists() && b.exists() && c.exists());
        assert!(!d.exists() && !old.exists());
        assert!(foreign.exists());
        super::prune_other_keys(&dir, "keccak", "keccak-k0", 1, now);
        assert!(own.exists() && !a.exists() && !b.exists() && !c.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The chunked digest changes with any byte, and with the length.
    #[test]
    fn chunked_digest_sensitivity() {
        let mut bytes: Vec<u8> = (0..(3 * super::DIGEST_CHUNK + 17))
            .map(|i| (i * 7) as u8)
            .collect();
        let d0 = super::chunked_digest(&bytes);
        assert_eq!(d0, super::chunked_digest(&bytes));
        for at in [
            0,
            super::DIGEST_CHUNK - 1,
            super::DIGEST_CHUNK,
            bytes.len() - 1,
        ] {
            bytes[at] ^= 1;
            assert_ne!(super::chunked_digest(&bytes), d0, "flip at {at}");
            bytes[at] ^= 1;
        }
        bytes.push(0);
        assert_ne!(super::chunked_digest(&bytes), d0);
        assert_eq!(super::chunked_digest(&[]).len(), 64);
    }

    /// K7: the CPU cost of the archive hashes a warm start pays (K8 finding 5), as an interleaved
    /// A/B: plain sha256 (K8's form, `RISC0_METAL_ARCHIVE_HASH=sha256`) and the chunked parallel
    /// digest, alternated `K7_PAIRS` times (default 11; A B A B ..., which goes first alternates)
    /// over the embedded metallib (the key's input) and over the newest keccak archive in
    /// `RISC0_METAL_ARCHIVE_DIR` / the cache dir (the checksum's input). Prints the medians,
    /// min/max and the load average before and after. The rest of the key (sha256 of < 1 KB of
    /// names) is the same in both forms.
    #[test]
    #[ignore]
    fn k7_archive_hash_cost() {
        use risc0_zkp::core::hash::sha::Sha256 as _;
        let pairs: usize = std::env::var("K7_PAIRS").map_or(11, |v| v.parse().unwrap());
        let loadavg = || {
            std::process::Command::new("sysctl")
                .args(["-n", "vm.loadavg"])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_default()
        };
        let time = |f: &dyn Fn() -> String| {
            let t0 = std::time::Instant::now();
            std::hint::black_box(f());
            t0.elapsed().as_secs_f64() * 1e3
        };
        let stats = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (v[v.len() / 2], v[0], v[v.len() - 1])
        };
        let ab = |what: &str, bytes: &[u8]| {
            let plain = || super::Sha256Impl::hash_bytes(bytes).to_string();
            let chunked = || super::parallel_digest(bytes);
            // One untimed call of each (page-in, thread start-up).
            plain();
            chunked();
            let before = loadavg();
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for i in 0..pairs {
                if i % 2 == 0 {
                    a.push(time(&plain));
                    b.push(time(&chunked));
                } else {
                    b.push(time(&chunked));
                    a.push(time(&plain));
                }
            }
            let ratios: Vec<f64> = b.iter().zip(&a).map(|(b, a)| b / a).collect();
            let (am, amin, amax) = stats(&mut a);
            let (bm, bmin, bmax) = stats(&mut b);
            let (rm, _, _) = stats(&mut ratios.clone());
            println!(
                "K7HASH {what} ({} MB), {pairs} pairs: plain sha256 median {am:.1} ms \
                 [{amin:.1}..{amax:.1}], chunked median {bm:.1} ms [{bmin:.1}..{bmax:.1}], \
                 per-pair chunked/plain median {rm:.3}; load {before} -> {}",
                bytes.len() >> 20,
                loadavg()
            );
        };
        ab("key input: embedded metallib", METAL_LIB);
        let dir = super::archive_dir().unwrap();
        let newest = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("keccak-"))
            .max_by_key(|e| e.metadata().unwrap().modified().unwrap());
        match newest {
            Some(e) => ab(
                "checksum input: keccak archive",
                &std::fs::read(e.path()).unwrap(),
            ),
            None => println!("K7HASH no keccak archive in {}", dir.display()),
        }
    }

    #[test]
    fn archive_key_inputs() {
        let base = archive_key("Apple M5 Max", "25A1", b"lib", &["a", "b"]);
        assert_eq!(base.len(), 24);
        assert_eq!(
            base,
            archive_key("Apple M5 Max", "25A1", b"lib", &["a", "b"])
        );
        for other in [
            archive_key("Apple M4", "25A1", b"lib", &["a", "b"]),
            archive_key("Apple M5 Max", "25A2", b"lib", &["a", "b"]),
            archive_key("Apple M5 Max", "25A1", b"lib2", &["a", "b"]),
            archive_key("Apple M5 Max", "25A1", b"lib", &["a"]),
            archive_key("Apple M5 Max", "25A1", b"lib", &["ab"]),
        ] {
            assert_ne!(base, other);
        }
        assert!(super::os_build().is_some_and(|os| !os.is_empty()));
    }

    /// An unknown OS build disables the archive instead of keying it without the OS (an OS
    /// update would otherwise keep the old archive). Before K8 the key used "unknown".
    #[test]
    fn unknown_os_build_disables_archive() {
        let dir = Path::new("/nonexistent");
        assert!(
            super::archive_base_for(dir, "Apple M5 Max", None, "keccak", b"lib", &["a"]).is_none()
        );
        let base =
            super::archive_base_for(dir, "Apple M5 Max", Some("25A1"), "keccak", b"lib", &["a"])
                .unwrap();
        assert_eq!(
            base,
            dir.join(format!(
                "keccak-{}",
                archive_key("Apple M5 Max", "25A1", b"lib", &["a"])
            ))
        );
    }

    /// Cold vs warm PSO creation of every keccak kernel (K5 measurement; `--ignored`). Three
    /// loads of the metallib in this process: (1) no archive (OS shader cache only: cold or warm
    /// depending on the machine's cache), (2) through a fresh archive (misses, then add +
    /// serialize), (3) through that archive (hits only). `K5_ARCHIVE_DIR` keeps the archive in a
    /// given directory across processes (run twice to time a hit in a fresh process).
    #[test]
    #[ignore]
    fn archive_cold_warm_profile() {
        let _gpu = super::super::tests::gpu_test_lock();
        let device = device();
        let names = super::super::kernel_names();
        let dir = std::env::var_os("K5_ARCHIVE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| temp_dir("profile"));
        let skip_plain = std::env::var("K5_SKIP_PLAIN").is_ok_and(|v| v == "1");
        let secs = |d: std::time::Duration| d.as_secs_f64();
        if !skip_plain {
            let t0 = std::time::Instant::now();
            let plain =
                ArchivedKernels::new_in(None, &device, "keccak", METAL_LIB, &names).unwrap();
            println!(
                "K5 plain (no archive): {} PSOs in {:.2} s",
                names.len(),
                secs(t0.elapsed())
            );
            drop(plain);
        }
        for run in ["archive run 1", "archive run 2"] {
            let k =
                ArchivedKernels::new_in(Some(dir.clone()), &device, "keccak", METAL_LIB, &names)
                    .unwrap();
            let r = &k.report;
            println!(
                "K5 {run}: {} PSOs in {:.2} s: loaded {} hits {} misses {} adds {:.2} s (summed over threads) saved {} (serialize {:.2} s) errors {:?} size {} B path {}",
                names.len(),
                secs(r.elapsed),
                r.loaded,
                r.hits,
                r.misses,
                secs(r.add_time),
                r.saved,
                secs(r.save_time),
                r.errors,
                r.path
                    .as_ref()
                    .and_then(|p| std::fs::metadata(p).ok())
                    .map_or(0, |m| m.len()),
                r.path.as_ref().unwrap().display(),
            );
        }
    }

    /// Probe of an archive file made elsewhere (K5's `metal-tt` evaluation; `--ignored`):
    /// `K5_PROBE_ARCHIVE=<file>` is loaded as a binary archive and every keccak PSO is created
    /// from it with `FailOnBinaryArchiveMiss`; prints hits/misses and the time.
    #[test]
    #[ignore]
    fn probe_archive_file() {
        let path = PathBuf::from(std::env::var("K5_PROBE_ARCHIVE").expect("K5_PROBE_ARCHIVE"));
        let _gpu = super::super::tests::gpu_test_lock();
        let device = device();
        let library = device.new_library_with_data(METAL_LIB).unwrap();
        let desc = metal::BinaryArchiveDescriptor::new();
        desc.set_url(&super::file_url(&path));
        let archive = match device.new_binary_archive_with_descriptor(&desc) {
            Ok(a) => a,
            Err(e) => panic!("K5 probe: loading {} failed: {e}", path.display()),
        };
        let t0 = std::time::Instant::now();
        let (mut hits, mut misses) = (Vec::new(), Vec::new());
        for name in super::super::kernel_names() {
            let f = library.get_function(name, None).unwrap();
            let d = metal::ComputePipelineDescriptor::new();
            d.set_compute_function(Some(&f));
            d.set_binary_archives(&[&archive]);
            match super::new_pso_with_options(
                &device,
                &d,
                metal::MTLPipelineOption::FailOnBinaryArchiveMiss,
            ) {
                Ok(_) => hits.push(name),
                Err(e) => misses.push(format!("{name}: {e}")),
            }
        }
        println!(
            "K5 probe {}: {} hits, {} misses in {:.2} s; first misses: {:?}",
            path.display(),
            hits.len(),
            misses.len(),
            t0.elapsed().as_secs_f64(),
            &misses[..misses.len().min(3)]
        );
    }
}
