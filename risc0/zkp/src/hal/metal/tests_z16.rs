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

/// Put `content` into `aot` as the offline archive of the zkp unit for this GPU.
fn place_aot(aot: &std::path::Path, arch: &str, content: &[u8]) -> std::path::PathBuf {
    let _ = std::fs::remove_dir_all(aot);
    std::fs::create_dir_all(aot).unwrap();
    let base = aot.join(archive::aot_stem(ZKP_ARCHIVE_UNIT, METAL_LIB, arch));
    let path = archive::archive_file(&base, content);
    std::fs::write(&path, content).unwrap();
    path
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
    println!(
        "Z16PROBE arch {arch} stem {}",
        archive::aot_stem(&unit, &lib, &arch)
    );
}
