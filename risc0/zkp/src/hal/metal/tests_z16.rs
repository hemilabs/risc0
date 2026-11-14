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

//! Z16: the zkp kernel set through the binary archive (`MetalKernels::new_zkp_in`).

use metal::Device;
use risc0_core::field::baby_bear::BabyBearElem;

use super::{
    archive, KernelArg, MetalHalSha256, MetalKernels, KERNEL_NAMES, METAL_LIB, ZKP_ARCHIVE_UNIT,
};
use crate::hal::{Buffer as _, Hal as _};

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("risc0-z16-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn archive_files(dir: &std::path::Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".binarchive"))
        .collect();
    v.sort();
    v
}

/// The archived loader writes a `zkp-` archive on a fresh directory and loads every PSO from it
/// on the next call (all hits, nothing saved); the plain loader (`RISC0_METAL_ARCHIVE_ZKP=0`)
/// builds the same kernel set without a report. A PSO from each loader computes the same result.
#[test]
fn zkp_kernels_archive_roundtrip() {
    let device = Device::system_default().unwrap();
    let dir = temp_dir("roundtrip");
    let (first, r1) =
        MetalKernels::new_zkp_in(&device, METAL_LIB, KERNEL_NAMES, Some(Some(dir.clone())))
            .unwrap();
    let r1 = r1.unwrap();
    if archive::os_build().is_none() {
        eprintln!("OS build unknown: no archive to test");
        return;
    }
    assert!(r1.errors.is_empty(), "{:?}", r1.errors);
    assert_eq!(r1.hits + r1.misses, KERNEL_NAMES.len());
    assert_eq!(
        r1.misses,
        KERNEL_NAMES.len(),
        "fresh directory: every PSO is a miss"
    );
    assert!(r1.saved);
    let files = archive_files(&dir);
    assert_eq!(files.len(), 1, "{files:?}");
    assert!(files[0].starts_with("zkp-"), "{files:?}");

    let (second, r2) =
        MetalKernels::new_zkp_in(&device, METAL_LIB, KERNEL_NAMES, Some(Some(dir.clone())))
            .unwrap();
    let r2 = r2.unwrap();
    assert!(r2.loaded && !r2.saved, "{r2:?}");
    assert_eq!(r2.hits, KERNEL_NAMES.len(), "{r2:?}");
    assert_eq!(archive_files(&dir), files);

    let (plain, r3) = MetalKernels::new_zkp_in(&device, METAL_LIB, KERNEL_NAMES, None).unwrap();
    assert!(r3.is_none());
    for k in [&first, &second, &plain] {
        let mut names: Vec<&str> = k.names().collect();
        names.sort();
        let mut want = KERNEL_NAMES.to_vec();
        want.sort();
        assert_eq!(names, want);
    }

    // The same kernel from the plain, the compiled-into-archive and the archive-hit PSO.
    let hal = MetalHalSha256::new();
    let n = 4099;
    let input: Vec<BabyBearElem> = (0..n as u32)
        .map(|i| BabyBearElem::new(i * 7919 + 3))
        .collect();
    let factor = [BabyBearElem::new(1234567)];
    let mut outs = Vec::new();
    for k in [&plain, &first, &second] {
        let io = hal.copy_from_elem("io", &input);
        hal.dispatch(
            k.pipeline("eltwise_mul_factor_fp"),
            &[io.as_arg(), KernelArg::Bytes(super::as_bytes(&factor))],
            super::Grid::Threads(n as u64),
            &[],
        );
        outs.push(io.to_vec());
    }
    let want: Vec<BabyBearElem> = input.iter().map(|x| *x * factor[0]).collect();
    for o in &outs {
        assert_eq!(o, &want);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The archived loader without a directory (`RISC0_METAL_ARCHIVE=0`) builds every PSO
/// concurrently and reports no archive.
#[test]
fn zkp_kernels_archive_off() {
    let device = Device::system_default().unwrap();
    let (k, r) = MetalKernels::new_zkp_in(&device, METAL_LIB, KERNEL_NAMES, Some(None)).unwrap();
    let r = r.unwrap();
    assert!(r.path.is_none() && !r.loaded && !r.saved, "{r:?}");
    assert_eq!(r.hits, 0);
    assert_eq!(r.misses, KERNEL_NAMES.len());
    assert!(KERNEL_NAMES.iter().all(|n| k.get(n).is_some()));
}

/// Z16 offline-archive probe (`--ignored`, manual): `Z16_PROBE_METALLIB=<file>` is loaded, then
/// `Z16_PROBE_ARCHIVE=<file>` (e.g. made by `metal-tt`) as a binary archive, and a PSO of every
/// `Z16_PROBE_NAMES` (comma-separated) kernel is created from it with
/// `FailOnBinaryArchiveMiss`; prints hits, misses and the time. Run with
/// `MTL_SHADER_CACHE_SIZE=0` to show that a hit needs no compile.
#[test]
#[ignore]
fn z16_probe_archive_file() {
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
    let lib = std::fs::read(var("Z16_PROBE_METALLIB")).unwrap();
    let path = std::path::PathBuf::from(var("Z16_PROBE_ARCHIVE"));
    let names = var("Z16_PROBE_NAMES");
    let device = Device::system_default().unwrap();
    let t0 = std::time::Instant::now();
    let library = device.new_library_with_data(&lib).unwrap();
    let desc = metal::BinaryArchiveDescriptor::new();
    desc.set_url(&archive::file_url(&path));
    let arch = device
        .new_binary_archive_with_descriptor(&desc)
        .unwrap_or_else(|e| panic!("loading {}: {e}", path.display()));
    let t1 = std::time::Instant::now();
    let (mut hits, mut misses) = (0, Vec::new());
    for name in names.split(',') {
        let f = library.get_function(name, None).unwrap();
        let d = metal::ComputePipelineDescriptor::new();
        d.set_compute_function(Some(&f));
        d.set_binary_archives(&[&arch]);
        match archive::new_pso_with_options(
            &device,
            &d,
            metal::MTLPipelineOption::FailOnBinaryArchiveMiss,
        ) {
            Ok(_) => hits += 1,
            Err(e) => misses.push(format!("{name}: {e}")),
        }
    }
    println!(
        "Z16PROBE archive {}: {hits} hits, {} misses; load {:.3} s, PSOs {:.3} s; misses: {:?}",
        path.display(),
        misses.len(),
        (t1 - t0).as_secs_f64(),
        t1.elapsed().as_secs_f64(),
        &misses[..misses.len().min(3)]
    );
}

fn only_archive(dir: &std::path::Path) -> std::path::PathBuf {
    let files = archive_files(dir);
    assert_eq!(files.len(), 1, "{files:?}");
    dir.join(&files[0])
}

/// This OS's first (exact-build) offline-archive OS key.
fn os_key() -> String {
    archive::aot_os_keys()
        .into_iter()
        .next()
        .expect("OS build known")
}

/// Put `content` into `aot` as the offline archive of the zkp unit for this GPU and OS key
/// `os` (the directory is emptied first).
fn place_aot_os(aot: &std::path::Path, arch: &str, os: &str, content: &[u8]) -> std::path::PathBuf {
    let _ = std::fs::remove_dir_all(aot);
    std::fs::create_dir_all(aot).unwrap();
    let stem = archive::aot_stem(ZKP_ARCHIVE_UNIT, METAL_LIB, arch, os);
    let path = aot.join(archive::aot_file_name(&stem, content));
    std::fs::write(&path, content).unwrap();
    path
}

/// [place_aot_os] with this OS's exact-build key.
fn place_aot(aot: &std::path::Path, arch: &str, content: &[u8]) -> std::path::PathBuf {
    place_aot_os(aot, arch, &os_key(), content)
}

/// Z16 offline archives (`RISC0_METAL_AOT_DIR`; a runtime archive of this machine stands in for a
/// `metal-tt` one, which the Z16 note tests by hand): a matching file serves every PSO
/// (`aot_hits`), and its PSOs compute the right result; a file whose content does not match its
/// checksum is skipped (reported, no crash) and the PSOs are built as without it; an archive
/// with only some kernels serves those and the rest are built; the offline dir is never written.
#[test]
fn zkp_kernels_aot_archive() {
    let device = Device::system_default().unwrap();
    let (Some(arch), Some(_)) = (archive::gpu_architecture(&device), archive::os_build()) else {
        eprintln!("no GPU architecture name / OS build: nothing to test");
        return;
    };
    let rt = temp_dir("aot-rt");
    let aot = temp_dir("aot");
    // A full archive of the zkp set, written by the runtime path.
    MetalKernels::new_zkp_in(&device, METAL_LIB, KERNEL_NAMES, Some(Some(rt.clone()))).unwrap();
    let full = std::fs::read(only_archive(&rt)).unwrap();

    let placed = place_aot(&aot, &arch, &full);
    let k = archive::ArchivedKernels::new_in_aot(
        None,
        Some(aot.clone()),
        &device,
        ZKP_ARCHIVE_UNIT,
        METAL_LIB,
        KERNEL_NAMES,
        None,
    )
    .unwrap();
    let r = &k.report;
    assert!(r.errors.is_empty(), "{:?}", r.errors);
    assert_eq!(r.aot_path.as_deref(), Some(placed.as_path()));
    assert_eq!(
        (r.aot_hits, r.hits, r.misses),
        (KERNEL_NAMES.len(), KERNEL_NAMES.len(), 0)
    );
    assert_eq!(archive_files(&aot).len(), 1, "offline dir unchanged");
    let hal = MetalHalSha256::new();
    let n = 1031;
    let input: Vec<BabyBearElem> = (0..n as u32)
        .map(|i| BabyBearElem::new(i * 31 + 5))
        .collect();
    let factor = [BabyBearElem::new(7654321)];
    let io = hal.copy_from_elem("io", &input);
    hal.dispatch(
        k.pipeline("eltwise_mul_factor_fp"),
        &[io.as_arg(), KernelArg::Bytes(super::as_bytes(&factor))],
        super::Grid::Threads(n as u64),
        &[],
    );
    let want: Vec<BabyBearElem> = input.iter().map(|x| *x * factor[0]).collect();
    assert_eq!(io.to_vec(), want);
    drop(k);

    // Damaged (content no longer matches the name): skipped before Metal sees it.
    let mut bad = full.clone();
    let mid = bad.len() / 2;
    bad[mid] ^= 0x5a;
    let placed = place_aot(&aot, &arch, &full);
    std::fs::write(&placed, &bad).unwrap();
    let k = archive::ArchivedKernels::new_in_aot(
        None,
        Some(aot.clone()),
        &device,
        ZKP_ARCHIVE_UNIT,
        METAL_LIB,
        KERNEL_NAMES,
        None,
    )
    .unwrap();
    assert!(k.report.aot_path.is_none());
    assert_eq!(k.report.aot_hits, 0);
    assert_eq!(k.report.errors.len(), 1, "{:?}", k.report.errors);
    assert!(KERNEL_NAMES.iter().all(|n| k.get(n).is_some()));
    assert!(placed.exists(), "the offline dir is never written");
    drop(k);

    // Another architecture's file is not used.
    place_aot(&aot, "applegpu_g0x", &full);
    let k = archive::ArchivedKernels::new_in_aot(
        None,
        Some(aot.clone()),
        &device,
        ZKP_ARCHIVE_UNIT,
        METAL_LIB,
        KERNEL_NAMES,
        None,
    )
    .unwrap();
    assert!(k.report.aot_path.is_none() && k.report.errors.is_empty());
    drop(k);

    // Partial: an archive of two kernels serves those two; the rest go through the runtime archive.
    let part = temp_dir("aot-part");
    let two = &KERNEL_NAMES[..2];
    archive::ArchivedKernels::new_in(
        Some(part.clone()),
        &device,
        ZKP_ARCHIVE_UNIT,
        METAL_LIB,
        two,
    )
    .unwrap();
    place_aot(&aot, &arch, &std::fs::read(only_archive(&part)).unwrap());
    let rt2 = temp_dir("aot-rt2");
    let k = archive::ArchivedKernels::new_in_aot(
        Some(rt2.clone()),
        Some(aot.clone()),
        &device,
        ZKP_ARCHIVE_UNIT,
        METAL_LIB,
        KERNEL_NAMES,
        None,
    )
    .unwrap();
    let r = &k.report;
    assert!(r.errors.is_empty(), "{:?}", r.errors);
    assert_eq!(r.aot_hits, 2);
    assert_eq!(r.misses, KERNEL_NAMES.len() - 2);
    assert!(r.saved);
    for d in [rt, aot, part, rt2] {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// Z16 (`--ignored`, manual): prints this GPU's architecture name and the offline-archive stem
/// of `Z16_PROBE_METALLIB` under unit `Z16_PROBE_UNIT` (checks `tools/metal_aot` naming).
#[test]
#[ignore]
fn z16_print_aot_stem() {
    let lib = std::fs::read(std::env::var("Z16_PROBE_METALLIB").unwrap()).unwrap();
    let unit = std::env::var("Z16_PROBE_UNIT").unwrap();
    let device = Device::system_default().unwrap();
    let arch = archive::gpu_architecture(&device).unwrap_or_default();
    for os in archive::aot_os_keys() {
        println!(
            "Z16PROBE arch {arch} stem {}",
            archive::aot_stem(&unit, &lib, &arch, &os)
        );
    }
}

/// A zkp archive of the whole kernel set, written by the runtime path into a fresh directory
/// (stands in for a `metal-tt` file; the W7-Z1 note tests real ones by hand).
fn runtime_zkp_archive(name: &str) -> Vec<u8> {
    let device = Device::system_default().unwrap();
    let rt = temp_dir(name);
    MetalKernels::new_zkp_in(&device, METAL_LIB, KERNEL_NAMES, Some(Some(rt.clone()))).unwrap();
    let full = std::fs::read(only_archive(&rt)).unwrap();
    let _ = std::fs::remove_dir_all(&rt);
    full
}

fn load_aot(rt: Option<std::path::PathBuf>, aot: &std::path::Path) -> archive::ArchivedKernels {
    let device = Device::system_default().unwrap();
    archive::ArchivedKernels::new_in_aot(
        rt,
        Some(aot.to_path_buf()),
        &device,
        ZKP_ARCHIVE_UNIT,
        METAL_LIB,
        KERNEL_NAMES,
        None,
    )
    .unwrap()
}

/// W7-Z1 (a): the OS key in the offline name. A file of another OS build, another major version
/// or the pre-W7-Z1 name without a key is not opened (a clean miss: no error, no AOT hit, the
/// PSOs are built as without it); the major-version key (`macos<N>`, the packaging form) is
/// used when there is no exact-build file, and the exact-build file wins over it.
#[test]
fn w7z1_aot_os_key() {
    let device = Device::system_default().unwrap();
    let (Some(arch), Some(_)) = (archive::gpu_architecture(&device), archive::os_build()) else {
        eprintln!("no GPU architecture name / OS build: nothing to test");
        return;
    };
    let keys = archive::aot_os_keys();
    assert_eq!(keys.len(), 2, "{keys:?}");
    assert!(
        keys[0].starts_with("os") && keys[1].starts_with("macos"),
        "{keys:?}"
    );
    let full = runtime_zkp_archive("w7z1-oskey-rt");
    let aot = temp_dir("w7z1-oskey");
    let n = KERNEL_NAMES.len();

    for other in ["os0A000", "macos1"] {
        place_aot_os(&aot, &arch, other, &full);
        let k = load_aot(None, &aot);
        let r = &k.report;
        assert!(
            r.aot_path.is_none() && r.errors.is_empty(),
            "{other}: {r:?}"
        );
        assert_eq!((r.aot_hits, r.misses), (0, n), "{other}");
    }
    // The Z16 name (no OS key) is not this OS's file either.
    let _ = std::fs::remove_dir_all(&aot);
    std::fs::create_dir_all(&aot).unwrap();
    let digest = archive::parallel_digest(METAL_LIB);
    let z16 = aot.join(format!(
        "{ZKP_ARCHIVE_UNIT}-{}-{arch}-{}.binarchive",
        &digest[..24],
        archive::aot_checksum(&full)
    ));
    std::fs::write(&z16, &full).unwrap();
    let k = load_aot(None, &aot);
    assert!(k.report.aot_path.is_none() && k.report.errors.is_empty());

    // Major-version key alone: used.
    let major = place_aot_os(&aot, &arch, &keys[1], &full);
    let k = load_aot(None, &aot);
    assert_eq!(k.report.aot_path.as_deref(), Some(major.as_path()));
    assert_eq!(k.report.aot_hits, n);
    assert_eq!(k.report.aot_bytes, full.len() as u64);
    // Both: the exact build wins.
    let stem = archive::aot_stem(ZKP_ARCHIVE_UNIT, METAL_LIB, &arch, &keys[0]);
    let exact = aot.join(archive::aot_file_name(&stem, &full));
    std::fs::write(&exact, &full).unwrap();
    let k = load_aot(None, &aot);
    assert_eq!(k.report.aot_path.as_deref(), Some(exact.as_path()));
    let _ = std::fs::remove_dir_all(&aot);
}

/// W7-Z1 (a): the OS keys from the OS build and version strings.
#[test]
fn w7z1_aot_os_keys_for() {
    assert_eq!(
        archive::aot_os_keys_for(Some("26B131"), Some("27.2.1")),
        ["os26B131", "macos27"]
    );
    assert_eq!(archive::aot_os_keys_for(None, Some("28.0")), ["macos28"]);
    assert_eq!(
        archive::aot_os_keys_for(Some("2-6"), None),
        Vec::<String>::new()
    );
    assert_eq!(
        archive::aot_os_keys_for(Some(""), Some("x.1")),
        Vec::<String>::new()
    );
}

/// W7-Z1 (c): offline names use the parallel digest in every hash mode (what
/// `tools/metal_aot/build_archive.py` writes), and (b): a load with a directory of its own never
/// reads the offline directory from the environment (only the process's archive directory is
/// a process-wide load).
#[test]
fn w7z1_aot_name_and_scope() {
    let digest = archive::parallel_digest(METAL_LIB);
    assert_eq!(
        archive::aot_stem("u", METAL_LIB, "applegpu_g0x", "os1A1"),
        format!("u-{}-applegpu_g0x-os1A1", &digest[..24])
    );
    let content = b"not an archive";
    assert_eq!(
        archive::aot_file_name("s", content),
        format!("s-{}.binarchive", &archive::parallel_digest(content)[..16])
    );
    let own = temp_dir("w7z1-scope");
    assert!(!archive::is_process_dir(&own));
    assert!(archive::process_aot_dir(Some(&own)).is_none());
    assert!(archive::process_aot_dir(None).is_none());
    let _ = std::fs::remove_dir_all(&own);
}

/// W7-Z1 (d): an offline archive that serves every PSO, with an empty runtime directory, is
/// copied into the runtime archive in the background; a later load without the offline
/// directory then hits every PSO from the runtime archive (no compile).
#[test]
fn w7z1_aot_seeds_runtime_archive() {
    let device = Device::system_default().unwrap();
    let (Some(arch), Some(_)) = (archive::gpu_architecture(&device), archive::os_build()) else {
        return;
    };
    let full = runtime_zkp_archive("w7z1-seed-src");
    let aot = temp_dir("w7z1-seed-aot");
    let rt = temp_dir("w7z1-seed-rt");
    place_aot(&aot, &arch, &full);
    let k = load_aot(Some(rt.clone()), &aot);
    let r = &k.report;
    assert_eq!(r.aot_hits, KERNEL_NAMES.len(), "{r:?}");
    assert!(r.seeding && !r.saved, "{r:?}");
    drop(k);
    let t = std::time::Instant::now();
    while archive_files(&rt).is_empty() && t.elapsed().as_secs() < 30 {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let seeded = only_archive(&rt);
    let content = std::fs::read(&seeded).unwrap();
    assert_eq!(content, full);
    assert!(seeded.to_string_lossy().ends_with(&format!(
        "-{}.binarchive",
        archive::archive_checksum(&content)
    )));
    let (_, r2) =
        MetalKernels::new_zkp_in(&device, METAL_LIB, KERNEL_NAMES, Some(Some(rt.clone()))).unwrap();
    let r2 = r2.unwrap();
    assert!(r2.loaded && !r2.saved && r2.aot_hits == 0, "{r2:?}");
    assert_eq!(r2.hits, KERNEL_NAMES.len());
    // A second seed of the same key writes nothing.
    let base =
        archive::archive_base(&rt, &device, ZKP_ARCHIVE_UNIT, METAL_LIB, KERNEL_NAMES).unwrap();
    let again = archive::seed_runtime_archive(&only_archive(&aot), &base, ZKP_ARCHIVE_UNIT);
    assert!(again.unwrap().is_none());
    assert_eq!(archive_files(&rt).len(), 1);
    for d in [aot, rt] {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// W7-Z1 GC over a fake directory: per unit the two most recently used keys stay, older keys'
/// files go unless used within the hour, temporary files of dead saves go after an hour, other
/// files (offline names, unknown names) stay; the build cache keeps two metallibs per stem and
/// deletes older ones unused for a day. A dry run deletes nothing and reports the same.
#[test]
fn w7z1_gc_fake_dir() {
    use std::time::{Duration, SystemTime};
    let root = temp_dir("w7z1-gc");
    let dir = root.join("metal-archives");
    std::fs::create_dir_all(&dir).unwrap();
    let now = SystemTime::now();
    let h = |n: u64| now - Duration::from_secs(n * 3600);
    let put = |d: &std::path::Path, name: &str, t: SystemTime| {
        let p = d.join(name);
        std::fs::write(&p, vec![0u8; 100]).unwrap();
        std::fs::File::options()
            .write(true)
            .open(&p)
            .unwrap()
            .set_modified(t)
            .unwrap();
    };
    let k = |c: char| std::iter::repeat_n(c, 24).collect::<String>();
    let s = |c: char| std::iter::repeat_n(c, 16).collect::<String>();
    // unit rv32im: keys d (used 10 min ago), a (20 min), b (30 min: evicted but recent), c (old).
    let min = |n: u64| now - Duration::from_secs(n * 60);
    put(
        &dir,
        &format!("rv32im-{}-{}.binarchive", k('a'), s('1')),
        min(20),
    );
    put(
        &dir,
        &format!("rv32im-{}-{}.binarchive", k('b'), s('2')),
        min(30),
    );
    put(
        &dir,
        &format!("rv32im-{}-{}.binarchive", k('c'), s('3')),
        h(50),
    );
    put(
        &dir,
        &format!("rv32im-{}-{}.binarchive", k('d'), s('4')),
        min(10),
    );
    // unit rv32im_fc262144: two keys, both kept.
    put(
        &dir,
        &format!("rv32im_fc262144-{}-{}.binarchive", k('e'), s('5')),
        h(100),
    );
    put(
        &dir,
        &format!("rv32im_fc262144-{}-{}.binarchive", k('f'), s('6')),
        h(200),
    );
    // A temporary file of a dead save (old) and of one in progress (new).
    put(&dir, &format!(".zkp-{}.binarchive.tmp123", k('a')), h(3));
    put(&dir, &format!(".zkp-{}.binarchive.tmp456", k('b')), now);
    // Not runtime archive names: an offline name, a stray file.
    put(
        &dir,
        &format!(
            "zkp-{}-applegpu_g17s-os26B131-{}.binarchive",
            k('a'),
            s('7')
        ),
        h(900),
    );
    put(&dir, "notes.txt", h(900));
    // Build cache: stem metal_kernels_zkp with three, a bare-hash stem with one.
    let m = |c: char| std::iter::repeat_n(c, 32).collect::<String>();
    put(
        &root,
        &format!("metal_kernels_zkp-{}.metallib", m('a')),
        h(30),
    );
    put(
        &root,
        &format!("metal_kernels_zkp-{}.metallib", m('b')),
        h(40),
    );
    put(
        &root,
        &format!("metal_kernels_zkp-{}.metallib", m('c')),
        h(50),
    );
    put(
        &root,
        &format!("metal_kernels_rv32im-{}.metallib", m('a')),
        h(10),
    );
    put(
        &root,
        &format!("metal_kernels_rv32im-{}.metallib", m('b')),
        h(11),
    );
    put(
        &root,
        &format!("metal_kernels_rv32im-{}.metallib", m('c')),
        h(12),
    );
    put(&root, &format!("{}.metallib", "0".repeat(64)), h(900));

    let want: Vec<String> = vec![
        format!(".zkp-{}.binarchive.tmp123", k('a')),
        format!("metal_kernels_zkp-{}.metallib", m('c')),
        format!("rv32im-{}-{}.binarchive", k('c'), s('3')),
    ];
    let names = |r: &archive::GcReport| {
        let mut v: Vec<String> = r
            .deleted
            .iter()
            .map(|(p, _)| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    };
    let dry = archive::gc_archives(&dir, Some(&root), now, 2, true);
    assert_eq!(names(&dry), want);
    assert_eq!(dry.bytes, 300);
    // rv32im key b was used 30 min ago, metal_kernels_rv32im c 12 h ago: both left.
    assert_eq!(dry.recent, 2);
    let count = |d: &std::path::Path| std::fs::read_dir(d).unwrap().count();
    let (before_dir, before_root) = (count(&dir), count(&root));
    assert_eq!(before_dir, 10);
    let r = archive::gc_archives(&dir, Some(&root), now, 2, false);
    assert_eq!(names(&r), want);
    assert_eq!(count(&dir), before_dir - 2);
    assert_eq!(count(&root), before_root - 1);
    // Idempotent.
    assert!(archive::gc_archives(&dir, Some(&root), now, 2, false)
        .deleted
        .is_empty());
    // keep = 1: only the newest key of each unit stays (rv32im a and b are recent), so only
    // rv32im_fc262144 key f goes.
    let r = archive::gc_archives(&dir, None, now, 1, false);
    assert_eq!(r.deleted.len(), 1, "{:?}", r.deleted);
    assert!(r.deleted[0].0.to_string_lossy().contains(&k('f')));
    let _ = std::fs::remove_dir_all(&root);
}

/// W7-Z1 manual dry run of the GC on this machine's real archive directory and build cache
/// (`--ignored`): prints every file it would delete and the total. Deletes nothing.
#[test]
#[ignore]
fn w7z1_gc_dry_run_here() {
    let Some(dir) = archive::archive_dir() else {
        println!("W7Z1GC archive disabled");
        return;
    };
    let cache = dir.parent().map(|p| p.to_path_buf());
    let r = archive::gc_archives(
        &dir,
        cache.as_deref(),
        std::time::SystemTime::now(),
        archive::DEFAULT_GC_KEEP,
        std::env::var("W7Z1_GC_APPLY").is_err(),
    );
    for (p, size) in &r.deleted {
        println!("W7Z1GC delete {size:>10} {}", p.display());
    }
    println!(
        "W7Z1GC total {} files {} bytes ({:.1} MB); kept {}, recent {}",
        r.deleted.len(),
        r.bytes,
        r.bytes as f64 / 1e6,
        r.kept,
        r.recent
    );
}

/// W7-Z1 fc go/no-go probe (`--ignored`, manual): loads `W7Z1_METALLIB`, then the archive
/// `W7Z1_ARCHIVE` (e.g. `metal-tt` output of a `specialized_functions` script), and creates the
/// PSO of `W7Z1_NAME` specialized with `W7Z1_FC` (`index=value,...`, uint constants) from it with
/// `FailOnBinaryArchiveMiss`. Prints hit or miss and the times.
#[test]
#[ignore]
fn w7z1_probe_fc_archive() {
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
    let lib = std::fs::read(var("W7Z1_METALLIB")).unwrap();
    let path = std::path::PathBuf::from(var("W7Z1_ARCHIVE"));
    let name = var("W7Z1_NAME");
    let fc: Vec<(u64, u32)> = var("W7Z1_FC")
        .split(',')
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap();
            (k.parse().unwrap(), v.parse().unwrap())
        })
        .collect();
    let device = Device::system_default().unwrap();
    let t0 = std::time::Instant::now();
    let library = device.new_library_with_data(&lib).unwrap();
    let desc = metal::BinaryArchiveDescriptor::new();
    desc.set_url(&archive::file_url(&path));
    let arch = device
        .new_binary_archive_with_descriptor(&desc)
        .unwrap_or_else(|e| panic!("loading {}: {e}", path.display()));
    let t1 = std::time::Instant::now();
    let values = metal::FunctionConstantValues::new();
    for (i, v) in &fc {
        values.set_constant_value_at_index(
            v as *const u32 as *const std::ffi::c_void,
            metal::MTLDataType::UInt,
            *i,
        );
    }
    let f = library.get_function(&name, Some(values)).unwrap();
    let t2 = std::time::Instant::now();
    let d = metal::ComputePipelineDescriptor::new();
    d.set_compute_function(Some(&f));
    d.set_binary_archives(&[&arch]);
    let res = archive::new_pso_with_options(
        &device,
        &d,
        metal::MTLPipelineOption::FailOnBinaryArchiveMiss,
    );
    println!(
        "W7Z1PROBE fc {name} {fc:?} archive {}: {}; load {:.3} s, specialize {:.3} s, PSO {:.3} s",
        path.display(),
        match &res {
            Ok(_) => "HIT".to_string(),
            Err(e) => format!("MISS ({e})"),
        },
        (t1 - t0).as_secs_f64(),
        (t2 - t1).as_secs_f64(),
        t2.elapsed().as_secs_f64(),
    );
    if let (Ok(pso), Ok(n)) = (&res, std::env::var("W7Z1_RUN_N")) {
        // Optional check for a tiny `io[i] = io[i] * C + i` kernel.
        let n: usize = n.parse().unwrap();
        let input: Vec<u32> = (0..n as u32).collect();
        let buf = device.new_buffer_with_data(
            input.as_ptr() as *const _,
            (n * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let q = device.new_command_queue();
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(&buf), 0);
        enc.dispatch_threads(
            metal::MTLSize::new(n as u64, 1, 1),
            metal::MTLSize::new(64.min(n as u64), 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let out = unsafe { std::slice::from_raw_parts(buf.contents() as *const u32, n) };
        let c = fc[0].1;
        let ok = out
            .iter()
            .enumerate()
            .all(|(i, &v)| v == (i as u32).wrapping_mul(c).wrapping_add(i as u32));
        println!(
            "W7Z1PROBE run n={n} results {}",
            if ok { "OK" } else { "WRONG" }
        );
    }
}

/// W7-Z1 (`--ignored`, manual; no GPU work beyond loading the library): writes the inputs of
/// `tools/metal_aot/build_all.py` for this crate into `W7Z1_DUMP`: the embedded metallib as
/// `zkp.metallib` and one JSON line per archive unit (`unit`, `metallib`, `names`, `fc`) appended to
/// `units.jsonl`.
#[test]
#[ignore]
fn w7z1_dump_aot_inputs() {
    let dir = std::path::PathBuf::from(std::env::var("W7Z1_DUMP").expect("W7Z1_DUMP"));
    std::fs::create_dir_all(&dir).unwrap();
    let lib: &[u8] = METAL_LIB;
    std::fs::write(dir.join("zkp.metallib"), lib).unwrap();
    let mut lines = String::new();
    let mut unit = |unit: &str, names: &[&str], fc: Option<String>| {
        let names: Vec<String> = names.iter().map(|n| format!("\"{n}\"")).collect();
        lines.push_str(&format!(
            "{{\"unit\": \"{unit}\", \"metallib\": \"zkp.metallib\", \"names\": [{}], \"fc\": {}}}\n",
            names.join(", "),
            fc.map_or("null".to_string(), |f| format!("\"{f}\""))
        ));
    };
    unit(ZKP_ARCHIVE_UNIT, KERNEL_NAMES, None);
    use std::io::Write as _;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("units.jsonl"))
        .unwrap()
        .write_all(lines.as_bytes())
        .unwrap();
}

/// W7-Z1 warm-start probe (`--ignored`, manual): the steps of a warm archived load of the
/// metallib `W7Z1_METALLIB` (kernels `W7Z1_NAMES`, comma-separated) against its runtime archive
/// file `W7Z1_ARCHIVE`, each timed, `W7Z1_REPS` times (default 5), then the same kernels the
/// plain way (the OS shader cache) for comparison. Prints one `W7Z1WARM` line per repetition.
#[test]
#[ignore]
fn w7z1_probe_warm_costs() {
    use std::time::Instant;
    let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k} not set"));
    let lib = std::fs::read(var("W7Z1_METALLIB")).unwrap();
    let path = std::path::PathBuf::from(var("W7Z1_ARCHIVE"));
    let names_s = var("W7Z1_NAMES");
    let names: Vec<&str> = names_s.split(',').collect();
    let reps: usize = std::env::var("W7Z1_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let device = Device::system_default().unwrap();
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    for rep in 0..reps {
        let t = Instant::now();
        let _ = archive::os_build();
        let t_os = ms(t);
        let t = Instant::now();
        let library = device.new_library_with_data(&lib).unwrap();
        let t_lib = ms(t);
        let t = Instant::now();
        let d = archive::chunked_digest(&lib);
        let t_digest = ms(t);
        let t = Instant::now();
        let content = std::fs::read(&path).unwrap();
        let t_read = ms(t);
        let t = Instant::now();
        let sum = archive::archive_checksum(&content);
        let t_sum = ms(t);
        drop(content);
        let t = Instant::now();
        let desc = metal::BinaryArchiveDescriptor::new();
        desc.set_url(&archive::file_url(&path));
        let arch = device.new_binary_archive_with_descriptor(&desc).unwrap();
        let t_open = ms(t);
        let t = Instant::now();
        let mut hits = 0;
        for name in &names {
            let f = library.get_function(name, None).unwrap();
            let pd = metal::ComputePipelineDescriptor::new();
            pd.set_compute_function(Some(&f));
            pd.set_binary_archives(&[&arch]);
            if archive::new_pso_with_options(
                &device,
                &pd,
                metal::MTLPipelineOption::FailOnBinaryArchiveMiss,
            )
            .is_ok()
            {
                hits += 1;
            }
        }
        let t_psos = ms(t);
        let t = Instant::now();
        for name in &names {
            let f = library.get_function(name, None).unwrap();
            device.new_compute_pipeline_state_with_function(&f).unwrap();
        }
        let t_plain = ms(t);
        println!(
            "W7Z1WARM rep {rep} os_build {t_os:.2} ms, library {t_lib:.2} ms, metallib digest \
             {t_digest:.2} ms ({} MB), archive read {t_read:.2} ms, checksum {t_sum:.2} ms ({sum}), \
             archive open {t_open:.2} ms, {hits}/{} archive PSOs {t_psos:.2} ms; plain PSOs \
             {t_plain:.2} ms",
            lib.len() / 1_000_000,
            names.len(),
        );
        drop(d);
    }
}
