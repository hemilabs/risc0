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

//! Item G10 tests: the no-copy zkey mapping.

use std::time::Instant;

use super::super::srs::Srs;
use super::*;

fn mib(b: u64) -> f64 {
    b as f64 / (1u64 << 20) as f64
}

fn mem_line(what: &str, dev: &DeviceRef) -> (u64, ProcessMemory) {
    let a = dev.current_allocated_size();
    let m = process_memory();
    eprintln!(
        "G10MEM {what}: allocated {:.1} MiB, footprint {:.1} MiB, resident {:.1} MiB, wired {:.1} MiB",
        mib(a),
        mib(m.phys_footprint),
        mib(m.resident),
        mib(m.wired)
    );
    (a, m)
}

/// A small file mapped at unaligned offsets reads back its bytes, through the host view and
/// through a GPU blit out of the no-copy buffer.
#[test]
fn file_map_reads_back() {
    let _lock = super::super::test_lock::shared();
    let dev = metal::Device::system_default().unwrap();
    let path = std::env::temp_dir().join(format!("g10_filemap_{}.bin", std::process::id()));
    let data: Vec<u8> = (0..200_000u32).map(|i| (i * 7 + i / 251) as u8).collect();
    std::fs::write(&path, &data).unwrap();
    let file = File::open(&path).unwrap();
    let q = dev.new_command_queue();
    for (off, len) in [(0u64, 16u64), (4, 70_000), (16_388, 100_000), (199_996, 4)] {
        let m = FileMap::map(&dev, &file, &path, off, len, "test", off % 8 == 4).unwrap();
        assert_eq!(m.bytes(), &data[off as usize..(off + len) as usize]);
        assert_eq!(m.inner_offset(), off % page_size());
        assert_eq!(m.mapped_len() % page_size(), 0);
        let out = dev.new_buffer(len, MTLResourceOptions::StorageModeShared);
        let cb = q.new_command_buffer();
        let blit = cb.new_blit_command_encoder();
        blit.copy_from_buffer(m.buffer(), m.inner_offset(), &out, 0, len);
        blit.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = unsafe { std::slice::from_raw_parts(out.contents() as *const u8, len as usize) };
        assert_eq!(
            got,
            &data[off as usize..(off + len) as usize],
            "gpu off {off}"
        );
    }
    assert!(FileMap::map(&dev, &file, &path, 199_000, 2_000, "past end", false).is_err());
    drop(file);
    let _ = std::fs::remove_file(&path);
}

/// G10 probe (ignored; needs the rzup key): load the zkey both ways, report time, device
/// allocation and the process' footprint, and check the mapped points equal the loaded ones.
#[test]
#[ignore]
fn g10_mapped_srs_probe() {
    let _lock = super::super::test_lock::exclusive();
    let Some(dir) = super::super::srs::tests::component_dir() else {
        return;
    };
    let path = dir.join("stark_verify_final.zkey");
    let dev = metal::Device::system_default().unwrap();
    let (a0, m0) = mem_line("start", &dev);
    let t = Instant::now();
    let mapped = Srs::load_with(&dev, &path, true).unwrap();
    let t_map = t.elapsed();
    let (a1, m1) = mem_line("mapped", &dev);
    // Touch every page from the host (what a GPU pass would fault in).
    let t = Instant::now();
    let mut x = 0u64;
    for b in [
        mapped.a.bytes(),
        mapped.b_g1.bytes(),
        mapped.b_g2.bytes(),
        mapped.c.bytes(),
        mapped.h.bytes(),
    ] {
        for chunk in b.chunks(16384) {
            x = x.wrapping_add(chunk[0] as u64);
        }
    }
    let t_touch = t.elapsed();
    let (_, m2) = mem_line("mapped + touched", &dev);
    let t = Instant::now();
    let loaded = Srs::load(&dev, &path).unwrap();
    let t_load = t.elapsed();
    let (a3, m3) = mem_line("mapped + loaded", &dev);
    assert_eq!(mapped.a.bytes(), loaded.a.bytes());
    assert_eq!(mapped.b_g1.bytes(), loaded.b_g1.bytes());
    assert_eq!(mapped.b_g2.bytes(), loaded.b_g2.bytes());
    assert_eq!(mapped.c.bytes(), loaded.c.bytes());
    assert_eq!(mapped.h.bytes(), loaded.h.bytes());
    assert!(mapped.a.get(0) == loaded.a.get(0));
    eprintln!(
        "G10PROBE map {t_map:?} touch {t_touch:?} load {t_load:?} ({x}); mapped: allocated +{:.1} MiB \
         footprint +{:.1} MiB resident +{:.1} (touched +{:.1}) MiB; loaded: allocated +{:.1} MiB \
         footprint +{:.1} MiB",
        mib(a1 - a0),
        mib(m1.phys_footprint.saturating_sub(m0.phys_footprint)),
        mib(m1.resident.saturating_sub(m0.resident)),
        mib(m2.resident.saturating_sub(m0.resident)),
        mib(a3 - a1),
        mib(m3.phys_footprint.saturating_sub(m2.phys_footprint)),
    );
    drop(loaded);
    drop(mapped);
    mem_line("dropped", &dev);
}
