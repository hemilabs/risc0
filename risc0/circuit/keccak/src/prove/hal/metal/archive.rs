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
//! The implementation moved to the zkp Metal HAL (`risc0_zkp::hal::metal::archive`, R14) so that
//! rv32im and recursion use the same archive code under their own unit names; keccak keeps its
//! unit name (`keccak`), so its archive files and keys are unchanged. This module re-exports it
//! and keeps the archive tests, which run against the keccak metallib.
//!
//! W7-Z1: the two-archive (runtime + offline) hook is the shared one: `keccak_kernels` and
//! `keccak_strict_kernels` load through `ArchivedKernels::new`, a process-wide load, so their
//! PSOs are looked up in the offline archives of units `keccak` / `keccakstrict` in
//! `RISC0_METAL_AOT_DIR` (`tools/metal_aot`) first; `aot_archive_hook` below tests it.

pub(crate) use risc0_zkp::hal::metal::archive::*;

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
            let plain = || risc0_zkp::core::hash::sha::cpu::Impl::hash_bytes(bytes).to_string();
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

    /// The key covers the metallib, the device, the OS build and the kernel names.
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
    /// W7-Z1 (e): the offline-archive hook on the keccak metallib. A runtime archive of the small
    /// set, placed as the `keccak` offline archive for this GPU and OS, serves every PSO; a load
    /// with a directory of its own does not read `RISC0_METAL_AOT_DIR` (only a process-wide one).
    #[test]
    fn aot_archive_hook() {
        let _gpu = super::super::tests::gpu_test_lock();
        let device = device();
        let (Some(arch), Some(os)) = (
            super::gpu_architecture(&device),
            super::aot_os_keys().into_iter().next(),
        ) else {
            return;
        };
        let (rt, aot) = (temp_dir("w7z1-rt"), temp_dir("w7z1-aot"));
        std::fs::create_dir_all(&aot).unwrap();
        let names = small_set();
        let first = ArchivedKernels::new_in(Some(rt.clone()), &device, "keccak", METAL_LIB, &names)
            .unwrap();
        assert_eq!(first.report.aot_hits, 0);
        let content = std::fs::read(first.report.path.as_ref().unwrap()).unwrap();
        let stem = super::aot_stem("keccak", METAL_LIB, &arch, &os);
        std::fs::write(aot.join(super::aot_file_name(&stem, &content)), &content).unwrap();
        let k = ArchivedKernels::new_in_aot(
            None,
            Some(aot.clone()),
            &device,
            "keccak",
            METAL_LIB,
            &names,
            None,
        )
        .unwrap();
        assert!(k.report.errors.is_empty(), "{:?}", k.report.errors);
        assert_eq!((k.report.aot_hits, k.report.misses), (names.len(), 0));
        assert!(names.iter().all(|n| k.get(n).is_some()));
        for d in [rt, aot] {
            let _ = std::fs::remove_dir_all(d);
        }
    }
    /// W7-Z1 (`--ignored`, manual; no GPU work beyond loading the library): writes the inputs of
    /// `tools/metal_aot/build_all.py` for this crate into `W7Z1_DUMP`: the embedded metallib as
    /// `keccak.metallib` and one JSON line per archive unit (`unit`, `metallib`, `names`, `fc`) appended to
    /// `units.jsonl`.
    #[test]
    #[ignore]
    fn w7z1_dump_aot_inputs() {
        let dir = std::path::PathBuf::from(std::env::var("W7Z1_DUMP").expect("W7Z1_DUMP"));
        std::fs::create_dir_all(&dir).unwrap();
        let lib: &[u8] = METAL_LIB;
        std::fs::write(dir.join("keccak.metallib"), lib).unwrap();
        let mut lines = String::new();
        let mut unit = |unit: &str, names: &[&str], fc: Option<String>| {
            let names: Vec<String> = names.iter().map(|n| format!("\"{n}\"")).collect();
            lines.push_str(&format!(
                "{{\"unit\": \"{unit}\", \"metallib\": \"keccak.metallib\", \"names\": [{}], \"fc\": {}}}\n",
                names.join(", "),
                fc.map_or("null".to_string(), |f| format!("\"{f}\""))
            ));
        };
        let names = super::super::kernel_names();
        unit("keccak", &names, None);
        unit(
            "keccakstrict",
            &risc0_circuit_keccak_sys::metal_eval::EVAL_STRICT_STAGE_NAMES,
            None,
        );
        use std::io::Write as _;
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("units.jsonl"))
            .unwrap()
            .write_all(lines.as_bytes())
            .unwrap();
    }
}
