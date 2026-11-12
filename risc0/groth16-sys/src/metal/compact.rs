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

//! Compact preprocessed coefficients and their disk cache (item G10; port of the CUDA prover's
//! cold path, `kernels/cuda/groth16_coeffs.cuh` `compact_load` and `disk_cache.cuh`).
//!
//! The 48-byte `coeff_t` of `preprocessed_coeffs.bin` repeats the row's output index `c` and the
//! matrix `m` in every coefficient, and the rzup key has only ~16K distinct values among its 29M
//! coefficients. The compact form keeps, per coefficient, `(s, value index)` (8 bytes), plus the
//! table of distinct values, each row's output index (`rowc`) and the row start indices: ~0.32 GB
//! instead of 1.44 GB. `witness_into_poly_compact` sums in the same order, so `a` and `b` are
//! bit-identical.
//!
//! Deriving it reads and validates the whole original once (~1-3 s), so it is cached on disk:
//! `<cache dir>/coeffs_compact_metal_<key>.bin`, the cache dir being the CUDA library's
//! (`RISC0_G16_CACHE_DIR`, else `$XDG_CACHE_HOME/risc0-groth16`, else `~/.cache/risc0-groth16`).
//! The file name and magic carry the backend tag `metal` (the layout differs from CUDA's
//! `coeffs_compact_<key>.bin`: every section starts on a 16 KiB boundary so it can be mapped as a
//! no-copy Metal buffer). The key hashes the format (tag, version, element sizes), 64 sampled 4 KiB
//! blocks of the original, and its size, mtime, inode and device, so a cache of another key file
//! or another build is never used. Every load checks the header, a checksum of the payload, and
//! the invariants the kernel relies on ([CompactCoeffs::validate]); a bad file is rebuilt.
//! `RISC0_G16_METAL_COEFF_CACHE=0` keeps the compact form in memory only (rebuilt per load).

use std::{
    collections::HashMap,
    fs::File,
    hash::{BuildHasherDefault, Hasher},
    io::Write as _,
    os::unix::fs::{FileExt as _, MetadataExt as _},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{bail, ensure, Context as _, Result};
use ark_bn254::Fr;
use ark_ff::{BigInt, PrimeField as _};
use metal::DeviceRef;

use super::{
    coeffs::{validate_part, Coeff},
    filemap::HostMap,
    srs::{limbs_le, read_exact_at_par, SharedBuf},
};

const MAGIC: [u8; 8] = *b"R0G16MCP";
/// Bump when the layout or meaning of the cache file changes.
const VERSION: u32 = 1;
/// The backend tag in the key and the file name.
const TAG: &[u8] = b"metal-compact";
/// Section alignment of the cache file (the Apple silicon page size; a multiple of 4 KiB pages).
const ALIGN: u64 = 16384;
const HEADER_WORDS: usize = 16;

/// Where the compact coefficients came from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum CompactSource {
    /// A valid cache file.
    #[default]
    Cache,
    /// Derived from the original and written to the cache.
    Built,
    /// Derived from the original; not cached (no cache directory, write failed, or disabled).
    BuiltUncached,
}

/// What one [CompactCoeffs::load] cost.
#[derive(Clone, Debug, Default)]
pub struct CompactStats {
    pub source: CompactSource,
    pub cache_path: Option<PathBuf>,
    /// Deriving the compact form (including validating the original), and writing the cache.
    pub build: Duration,
    pub write: Duration,
    /// Mapping or reading, checksum and validation of the compact form.
    pub load: Duration,
    pub total: Duration,
    /// Payload bytes of the compact form.
    pub bytes: u64,
}

/// The compact coefficients in shared Metal buffers (mapped from the cache file, or allocated).
pub struct CompactCoeffs {
    /// `(s, value index)` per coefficient (2 u32 each): A's `count_a`, then B's `count_b`.
    pub cs: SharedBuf<u32>,
    /// The distinct values (8 u32 each, Fr in Montgomery form).
    pub table: SharedBuf<u32>,
    /// Output index of each row: A's rows at `[0, ind_a - 1)`, B's at `[ind_a, ind_a + ind_b - 1)`
    /// (the entry after each list's last row is unused).
    pub rowc: SharedBuf<u32>,
    /// Row starts as in [super::coeffs::Coeffs::indices].
    pub indices: SharedBuf<u32>,
    pub count_a: usize,
    pub count_b: usize,
    pub ind_a: usize,
    pub ind_b: usize,
    pub ntab: usize,
    pub stats: CompactStats,
}

/// The four arrays of the compact form (host copy, while building).
struct Built {
    cs: Vec<u32>,
    table: Vec<u32>,
    rowc: Vec<u32>,
    indices: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Header {
    key: u64,
    ca: u64,
    cb: u64,
    ua: u64,
    ub: u64,
    ntab: u64,
    sum: u64,
    off: [u64; 4],
    file_len: u64,
}

impl Header {
    fn lens(&self) -> [u64; 4] {
        let n = self.ca + self.cb;
        let u = self.ua + self.ub;
        [n * 8, self.ntab * 32, u * 4, u * 4]
    }

    fn layout(key: u64, ca: u64, cb: u64, ua: u64, ub: u64, ntab: u64) -> Self {
        let mut h = Self {
            key,
            ca,
            cb,
            ua,
            ub,
            ntab,
            sum: 0,
            off: [0; 4],
            file_len: 0,
        };
        let mut at = ALIGN;
        let lens = h.lens();
        for (o, len) in h.off.iter_mut().zip(lens) {
            *o = at;
            at = (at + len).div_ceil(ALIGN) * ALIGN;
        }
        h.file_len = h.off[3] + lens[3];
        h
    }

    fn to_bytes(self) -> [u8; HEADER_WORDS * 8] {
        let mut w = [0u64; HEADER_WORDS];
        w[0] = u64::from_le_bytes(MAGIC);
        w[1] = VERSION as u64 | (32u64 << 32);
        w[2..9].copy_from_slice(&[
            self.key, self.ca, self.cb, self.ua, self.ub, self.ntab, self.sum,
        ]);
        w[9..13].copy_from_slice(&self.off);
        w[13] = self.file_len;
        let mut b = [0u8; HEADER_WORDS * 8];
        for (c, v) in b.chunks_exact_mut(8).zip(w) {
            c.copy_from_slice(&v.to_le_bytes());
        }
        b
    }

    fn from_bytes(b: &[u8; HEADER_WORDS * 8]) -> Option<Self> {
        let w: Vec<u64> = b
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if w[0] != u64::from_le_bytes(MAGIC) || w[1] != VERSION as u64 | (32u64 << 32) {
            return None;
        }
        Some(Self {
            key: w[2],
            ca: w[3],
            cb: w[4],
            ua: w[5],
            ub: w[6],
            ntab: w[7],
            sum: w[8],
            off: [w[9], w[10], w[11], w[12]],
            file_len: w[13],
        })
    }
}

const FNV_BASIS: u64 = 0xcbf29ce484222325;

fn fnv1a(mut h: u64, b: &[u8]) -> u64 {
    for &x in b {
        h ^= x as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Checksum of a large payload: a multiply-xor over 64-bit words in 8 chunks on their own
/// threads, combined in order (the CUDA cache's `payload_hash`).
fn payload_hash(p: &[u8]) -> u64 {
    const CHUNKS: usize = 8;
    let words = p.len() / 8;
    let per = words.div_ceil(CHUNKS);
    let mut part = [0u64; CHUNKS];
    std::thread::scope(|s| {
        for (c, out) in part.iter_mut().enumerate() {
            s.spawn(move || {
                let mut h = 0x9e3779b97f4a7c15u64 ^ c as u64;
                let lo = (c * per).min(words);
                let hi = (lo + per).min(words);
                for w in p[lo * 8..hi * 8].chunks_exact(8) {
                    h = (h ^ u64::from_le_bytes(w.try_into().unwrap()))
                        .wrapping_mul(0xff51afd7ed558ccd);
                    h ^= h >> 32;
                }
                *out = h;
            });
        }
    });
    let mut bytes = Vec::with_capacity(CHUNKS * 8);
    for v in part {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let h = fnv1a(FNV_BASIS, &bytes);
    let h = fnv1a(h, &p[words * 8..]);
    fnv1a(h, &(p.len() as u64).to_le_bytes())
}

fn sections_hash(parts: [&[u8]; 4]) -> u64 {
    let mut h = FNV_BASIS;
    for p in parts {
        h = fnv1a(h, &payload_hash(p).to_le_bytes());
    }
    h
}

/// The Groth16 derived-data cache directory (created): `RISC0_G16_CACHE_DIR`, else
/// `$XDG_CACHE_HOME/risc0-groth16`, else `$HOME/.cache/risc0-groth16`; `None` if none can be
/// determined or created.
pub fn cache_dir() -> Option<PathBuf> {
    let d = std::env::var_os("RISC0_G16_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_CACHE_HOME").map(|x| PathBuf::from(x).join("risc0-groth16"))
        })
        .or_else(|| {
            std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/risc0-groth16"))
        })?;
    std::fs::create_dir_all(&d).ok()?;
    Some(d)
}

/// The cache files of this module in `d` (`coeffs_compact_metal_*.bin`, and temporaries).
fn cache_files(d: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(d) else {
        return Vec::new();
    };
    rd.filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("coeffs_compact_metal_"))
        })
        .collect()
}

/// Delete every Metal compact-coefficient cache file (the next load rebuilds it). Returns the
/// number of files removed.
pub fn invalidate_cache() -> usize {
    cache_dir().map_or(0, |d| invalidate_cache_in(&d))
}

/// [invalidate_cache] in `d`.
pub fn invalidate_cache_in(d: &Path) -> usize {
    cache_files(d)
        .into_iter()
        .filter(|p| std::fs::remove_file(p).is_ok())
        .count()
}

/// The disk cache is on unless `RISC0_G16_METAL_COEFF_CACHE=0`.
fn cache_enabled() -> bool {
    std::env::var("RISC0_G16_METAL_COEFF_CACHE").as_deref() != Ok("0")
}

/// Hasher for the 32-byte values while deriving the table (the values are field elements, so
/// a multiply-fold of the words is enough).
#[derive(Default)]
struct ValueHasher(u64);

impl Hasher for ValueHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            self.write_u64(u64::from_le_bytes(w));
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x9e3779b97f4a7c15);
    }
}

impl CompactCoeffs {
    /// Load the compact form of `preprocessed_coeffs.bin` at `path` (from the cache, or derived
    /// and cached), validated for `witness_len` / `domain_size` (the same checks as
    /// [super::coeffs::Coeffs::validate]: on a build the original is validated in full; a cache
    /// file is validated in its compact form). `mapped`: bind the cache file's sections as
    /// no-copy buffers ([SharedBuf::map]) instead of copying them into allocated buffers.
    pub fn load(
        device: &DeviceRef,
        path: &Path,
        witness_len: usize,
        domain_size: usize,
        mapped: bool,
    ) -> Result<Self> {
        let dir = if cache_enabled() { cache_dir() } else { None };
        Self::load_in(
            device,
            path,
            witness_len,
            domain_size,
            mapped,
            dir.as_deref(),
        )
    }

    /// [Self::load] with the cache in `cache_dir` (`None`: no disk cache).
    pub fn load_in(
        device: &DeviceRef,
        path: &Path,
        witness_len: usize,
        domain_size: usize,
        mapped: bool,
        cache_dir: Option<&Path>,
    ) -> Result<Self> {
        let t0 = Instant::now();
        let p = path.display();
        let file = File::open(path).with_context(|| format!("open(\"{p}\")"))?;
        let meta = file.metadata().with_context(|| format!("fstat(\"{p}\")"))?;
        let size = meta.len();
        ensure!(size >= 32, "\"{p}\" is truncated");
        let mut hdr = [0u8; 32];
        file.read_exact_at(&mut hdr, 0)
            .with_context(|| format!("reading \"{p}\""))?;
        let [ca, cb, ua, ub] = limbs_le(&hdr);
        let expect = (ca as u128 + cb as u128) * 48 + (ua as u128 + ub as u128) * 4 + 32;
        ensure!(
            expect == size as u128,
            "\"{p}\": header (coefficients {ca} + {cb}, indices {ua} + {ub}) needs {expect} \
             bytes, the file has {size}"
        );
        ensure!(
            ua >= 1 && ub >= 1,
            "\"{p}\": each index list needs at least its terminating count"
        );
        ensure!(
            ca + cb <= u32::MAX as u64,
            "\"{p}\": too many coefficients for the compact form"
        );
        let key = source_key(&file, &meta)?;
        let cache_path = cache_dir.map(|d| d.join(format!("coeffs_compact_metal_{key:016x}.bin")));
        let verbose = std::env::var_os("RISC0_VERBOSE").is_some();
        let mut stats = CompactStats {
            cache_path: cache_path.clone(),
            ..Default::default()
        };
        if let Some(cp) = &cache_path {
            if cp.is_file() {
                let t = Instant::now();
                match Self::from_cache(
                    device,
                    cp,
                    key,
                    [ca, cb, ua, ub],
                    witness_len,
                    domain_size,
                    mapped,
                ) {
                    Ok(mut c) => {
                        stats.load = t.elapsed();
                        stats.total = t0.elapsed();
                        stats.bytes = c.payload_bytes();
                        c.stats = stats;
                        return Ok(c);
                    }
                    Err(e) => {
                        if verbose {
                            eprintln!(
                                "[groth16-metal] rebuilding the compact coefficients cache {}: {e:#}",
                                cp.display()
                            );
                        }
                        // Not removed here: the rebuild below fails the same way on a sizes
                        // mismatch, and a successful rebuild renames over it.
                    }
                }
            }
        }

        // Build from the original.
        let t = Instant::now();
        let built = build(&file, path, [ca, cb, ua, ub], witness_len, domain_size)?;
        stats.build = t.elapsed();
        let ntab = built.table.len() / 8;
        if let Some(cp) = &cache_path {
            let t = Instant::now();
            let written = write_cache(cp, key, [ca, cb, ua, ub], &built);
            stats.write = t.elapsed();
            match written {
                Ok(()) => {
                    let t = Instant::now();
                    let mut c = Self::from_cache(
                        device,
                        cp,
                        key,
                        [ca, cb, ua, ub],
                        witness_len,
                        domain_size,
                        mapped,
                    )
                    .with_context(|| format!("reading back {}", cp.display()))?;
                    stats.load = t.elapsed();
                    stats.source = CompactSource::Built;
                    stats.total = t0.elapsed();
                    stats.bytes = c.payload_bytes();
                    c.stats = stats;
                    return Ok(c);
                }
                Err(e) => {
                    if verbose {
                        eprintln!(
                            "[groth16-metal] cannot write the compact coefficients cache {}: {e:#}",
                            cp.display()
                        );
                    }
                }
            }
        }
        let t = Instant::now();
        let to_buf = |v: &[u32], what: &str| -> Result<SharedBuf<u32>> {
            let mut b = SharedBuf::<u32>::alloc(device, what, v.len())?;
            b.as_bytes_mut().copy_from_slice(u32_bytes(v));
            Ok(b)
        };
        let mut c = Self {
            cs: to_buf(&built.cs, "compact coefficients")?,
            table: to_buf(&built.table, "compact coefficient values")?,
            rowc: to_buf(&built.rowc, "compact coefficient rows")?,
            indices: to_buf(&built.indices, "coefficient indices")?,
            count_a: ca as usize,
            count_b: cb as usize,
            ind_a: ua as usize,
            ind_b: ub as usize,
            ntab,
            stats: CompactStats::default(),
        };
        stats.load = t.elapsed();
        stats.source = CompactSource::BuiltUncached;
        stats.total = t0.elapsed();
        stats.bytes = c.payload_bytes();
        c.stats = stats;
        Ok(c)
    }

    fn from_cache(
        device: &DeviceRef,
        cp: &Path,
        key: u64,
        counts: [u64; 4],
        witness_len: usize,
        domain_size: usize,
        mapped: bool,
    ) -> Result<Self> {
        let [ca, cb, ua, ub] = counts;
        let file = File::open(cp).with_context(|| format!("open(\"{}\")", cp.display()))?;
        let size = file.metadata()?.len();
        let mut hb = [0u8; HEADER_WORDS * 8];
        file.read_exact_at(&mut hb, 0)?;
        let Some(h) = Header::from_bytes(&hb) else {
            bail!("bad magic or version");
        };
        let expect = Header::layout(key, ca, cb, ua, ub, h.ntab);
        ensure!(
            h.key == key && h.ntab >= 1 && h.ntab <= ca + cb && h.off == expect.off,
            "header does not match the source file"
        );
        ensure!(
            h.ca == ca && h.cb == cb && h.ua == ua && h.ub == ub,
            "header counts do not match the source file"
        );
        ensure!(
            h.file_len == expect.file_len && size == h.file_len,
            "file length {size} does not match the header ({})",
            h.file_len
        );
        let lens = h.lens();
        let sec = |i: usize, what: &str| -> Result<SharedBuf<u32>> {
            let words = (lens[i] / 4) as usize;
            if mapped {
                SharedBuf::<u32>::map(device, &file, cp, h.off[i], words, what)
            } else {
                let mut b = SharedBuf::<u32>::alloc(device, what, words)?;
                read_exact_at_par(&file, h.off[i], b.as_bytes_mut())
                    .with_context(|| format!("reading {what} from {}", cp.display()))?;
                Ok(b)
            }
        };
        let c = Self {
            cs: sec(0, "compact coefficients")?,
            table: sec(1, "compact coefficient values")?,
            rowc: sec(2, "compact coefficient rows")?,
            indices: sec(3, "coefficient indices")?,
            count_a: ca as usize,
            count_b: cb as usize,
            ind_a: ua as usize,
            ind_b: ub as usize,
            ntab: h.ntab as usize,
            stats: CompactStats::default(),
        };
        let sum = sections_hash([
            c.cs.bytes(),
            c.table.bytes(),
            c.rowc.bytes(),
            c.indices.bytes(),
        ]);
        ensure!(sum == h.sum, "checksum mismatch (damaged file)");
        c.validate(witness_len, domain_size)?;
        Ok(c)
    }

    /// Payload bytes of the four arrays.
    pub fn payload_bytes(&self) -> u64 {
        self.cs.byte_len() + self.table.byte_len() + self.rowc.byte_len() + self.indices.byte_len()
    }

    /// A u32 array's elements (any alignment).
    fn words(b: &SharedBuf<u32>) -> Vec<u32> {
        u32s(b.bytes())
    }

    /// What `witness_into_poly_compact` relies on: each index list starts at 0, increases
    /// strictly and ends at its coefficient count; each row's output index is below
    /// `domain_size` and increases strictly from row to row; every `s` is below `witness_len` and
    /// every value index below the table length; every table value is a canonical Fr.
    pub fn validate(&self, witness_len: usize, domain_size: usize) -> Result<()> {
        let ind = Self::words(&self.indices);
        let rowc = Self::words(&self.rowc);
        let (ua, ub) = (self.ind_a, self.ind_b);
        ensure!(
            ind.len() == ua + ub && rowc.len() == ua + ub,
            "index list lengths"
        );
        for (name, base, count, u) in [("A", 0usize, self.count_a, ua), ("B", ua, self.count_b, ub)]
        {
            let ix = &ind[base..base + u];
            ensure!(
                ix.first() == Some(&0) && ix.last().map(|&l| l as usize) == Some(count),
                "compact coefficients {name}: indices must run from 0 to {count}"
            );
            let rows = u - 1;
            if let Some(r) = (0..rows).find(|&r| ix[r] >= ix[r + 1]) {
                bail!("compact coefficients {name}: row {r} is empty or indices decrease");
            }
            let rc = &rowc[base..base + rows];
            if let Some(r) =
                (0..rows).find(|&r| rc[r] as usize >= domain_size || (r > 0 && rc[r] <= rc[r - 1]))
            {
                bail!(
                    "compact coefficients {name}: row {r} output {} out of range or not increasing",
                    rc[r]
                );
            }
        }
        let cs = self.cs.bytes();
        let ntab = self.ntab as u32;
        let wn = witness_len as u64;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .clamp(1, 16);
        let per = (cs.len() / 8).div_ceil(threads).max(1) * 8;
        let bad = std::thread::scope(|s| {
            let hs: Vec<_> = cs
                .chunks(per)
                .map(|part| {
                    s.spawn(move || {
                        part.chunks_exact(8).position(|e| {
                            let sig = u32::from_le_bytes(e[..4].try_into().unwrap());
                            let vi = u32::from_le_bytes(e[4..].try_into().unwrap());
                            sig as u64 >= wn || vi >= ntab
                        })
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().expect("validation thread panicked"))
                .collect::<Vec<_>>()
        });
        if let Some((part, pos)) = bad.iter().enumerate().find_map(|(i, b)| b.map(|p| (i, p))) {
            bail!(
                "compact coefficients: coefficient {} has s >= {witness_len} or a value index >= {ntab}",
                part * per / 8 + pos
            );
        }
        let r_mod = Fr::MODULUS;
        for (i, v) in self.table.bytes().chunks_exact(32).enumerate() {
            ensure!(
                BigInt::new(limbs_le(v)) < r_mod,
                "compact coefficients: table value {i} is not a canonical scalar"
            );
        }
        Ok(())
    }
}

fn u32_bytes(v: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding; the byte view covers exactly the slice.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn u32s(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// The cache key of the original coefficients file (see the module docs).
fn source_key(file: &File, meta: &std::fs::Metadata) -> Result<u64> {
    let size = meta.len();
    let mut h = fnv1a(FNV_BASIS, TAG);
    h = fnv1a(h, &VERSION.to_le_bytes());
    h = fnv1a(h, &32u32.to_le_bytes());
    h = fnv1a(h, &48u32.to_le_bytes());
    let mut block = vec![0u8; 4096];
    if size >= 4096 {
        for k in 0..64u64 {
            file.read_exact_at(&mut block, (size - 4096) / 63 * k)?;
            h = fnv1a(h, &block);
        }
    }
    for v in [
        size,
        meta.mtime() as u64,
        meta.mtime_nsec() as u64,
        meta.ino(),
        meta.dev(),
    ] {
        h = fnv1a(h, &v.to_le_bytes());
    }
    Ok(h)
}

/// Derive the compact form from the original (mapped, validated in full first).
fn build(
    file: &File,
    path: &Path,
    counts: [u64; 4],
    witness_len: usize,
    domain_size: usize,
) -> Result<Built> {
    let [ca, cb, ua, ub] = counts.map(|v| v as usize);
    let map = HostMap::map(file, path)?;
    let bytes = map.bytes();
    let n = ca + cb;
    let co_bytes = &bytes[32..32 + n * 48];
    // SAFETY: the mapping is page aligned, so +32 is 8-aligned (Coeff's alignment); Coeff is
    // Pod-like (repr(C) u32/u64 fields); the range is inside the file (size checked).
    let co: &[Coeff] = unsafe { std::slice::from_raw_parts(co_bytes.as_ptr() as *const Coeff, n) };
    let indices = u32s(&bytes[32 + n * 48..32 + n * 48 + (ua + ub) * 4]);
    validate_part("A", 0, &co[..ca], &indices[..ua], witness_len, domain_size)?;
    validate_part("B", 1, &co[ca..], &indices[ua..], witness_len, domain_size)?;

    let mut map_v: HashMap<[u64; 4], u32, BuildHasherDefault<ValueHasher>> = HashMap::default();
    let mut table: Vec<u32> = Vec::new();
    let mut cs = vec![0u32; 2 * n];
    for (i, c) in co.iter().enumerate() {
        let next = map_v.len() as u32;
        let vi = *map_v.entry(c.value).or_insert_with(|| {
            for l in c.value {
                table.push(l as u32);
                table.push((l >> 32) as u32);
            }
            next
        });
        cs[2 * i] = c.s;
        cs[2 * i + 1] = vi;
    }
    let mut rowc = vec![0u32; ua + ub];
    for r in 0..ua - 1 {
        rowc[r] = co[indices[r] as usize].c;
    }
    for r in 0..ub - 1 {
        rowc[ua + r] = co[ca + indices[ua + r] as usize].c;
    }
    Ok(Built {
        cs,
        table,
        rowc,
        indices,
    })
}

/// Write the cache file (temporary name, then renamed over `cp`).
fn write_cache(cp: &Path, key: u64, counts: [u64; 4], b: &Built) -> Result<()> {
    let [ca, cb, ua, ub] = counts;
    let mut h = Header::layout(key, ca, cb, ua, ub, (b.table.len() / 8) as u64);
    let parts = [
        u32_bytes(&b.cs),
        u32_bytes(&b.table),
        u32_bytes(&b.rowc),
        u32_bytes(&b.indices),
    ];
    h.sum = sections_hash(parts);
    let tmp = cp.with_extension(format!("tmp{}", std::process::id()));
    let res = (|| -> Result<()> {
        let mut f = std::io::BufWriter::with_capacity(
            8 << 20,
            File::create(&tmp).with_context(|| format!("create(\"{}\")", tmp.display()))?,
        );
        let mut at = 0u64;
        let hb = h.to_bytes();
        f.write_all(&hb)?;
        at += hb.len() as u64;
        for (i, part) in parts.iter().enumerate() {
            let pad = h.off[i] - at;
            f.write_all(&vec![0u8; pad as usize])?;
            f.write_all(part)?;
            at = h.off[i] + part.len() as u64;
        }
        ensure!(at == h.file_len, "cache layout");
        f.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        std::fs::rename(&tmp, cp)?;
        Ok(())
    })();
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::{
        coeffs::{tests::synthetic, Coeffs},
        prover::kernels,
        srs::tests::temp_path,
    };
    use metal::MTLResourceOptions;

    /// Runs the full and the compact witness_into_poly for A and B over a random witness;
    /// returns (full out, compact out, fault words).
    fn both_outputs(
        full: &Coeffs,
        cc: &CompactCoeffs,
        wn: usize,
        n: usize,
    ) -> (Vec<u8>, Vec<u8>, [u32; 2]) {
        let k = kernels().unwrap();
        let dev = &k.device;
        let mut wit = vec![0u8; wn * 32];
        for (i, e) in wit.chunks_exact_mut(32).enumerate() {
            let v = Fr::from((i as u64 + 3) * 0x9e37_79b9);
            e.copy_from_slice(
                &crate::metal::small::fr_to_words(&v)
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
        }
        let w = dev.new_buffer_with_data(
            wit.as_ptr().cast(),
            wit.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let zeros = vec![0u8; 2 * n * 32];
        let out_full = dev.new_buffer_with_data(
            zeros.as_ptr().cast(),
            zeros.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_cmp = dev.new_buffer_with_data(
            zeros.as_ptr().cast(),
            zeros.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let f0 = dev.new_buffer(16, MTLResourceOptions::StorageModeShared);
        let f1 = dev.new_buffer(16, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::write_bytes(f0.contents() as *mut u8, 0, 16);
            std::ptr::write_bytes(f1.contents() as *mut u8, 0, 16);
        }
        let cb = k.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let nn = n as u64;
        use crate::metal::prover::tests_support::{encode_witness_poly, CoeffsRef};
        encode_witness_poly(
            &k.small,
            enc,
            CoeffsRef::Full(full),
            &out_full,
            nn,
            &w,
            wn as u64,
            &f0,
        );
        encode_witness_poly(
            &k.small,
            enc,
            CoeffsRef::Compact(cc),
            &out_cmp,
            nn,
            &w,
            wn as u64,
            &f1,
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let read = |b: &metal::Buffer, len: usize| unsafe {
            std::slice::from_raw_parts(b.contents() as *const u8, len).to_vec()
        };
        (
            read(&out_full, zeros.len()),
            read(&out_cmp, zeros.len()),
            [read(&f0, 4)[0] as u32, read(&f1, 4)[0] as u32],
        )
    }

    #[test]
    fn compact_matches_full_and_caches() {
        let _lock = crate::metal::test_lock::shared();
        let dev = kernels().unwrap().device.clone();
        let (wn, n) = (40usize, 16usize);
        let (bytes, _, _) = synthetic(
            &[(0, 3), (2, 1), (7, 5), (9, 2), (15, 4)],
            &[(1, 2), (2, 4), (3, 1), (14, 6)],
            wn as u32,
        );
        // Make values repeat (the table must dedupe): copy coefficient 0's value into 5 others.
        let mut bytes = bytes;
        let v0: Vec<u8> = bytes[32 + 16..32 + 48].to_vec();
        for j in [2usize, 4, 9, 13, 20] {
            bytes[32 + j * 48 + 16..32 + j * 48 + 48].copy_from_slice(&v0);
        }
        let src = temp_path("compact_src.bin");
        std::fs::write(&src, &bytes).unwrap();
        let dir = temp_path("compact_cache");
        std::fs::create_dir_all(&dir).unwrap();

        let full = Coeffs::load(&dev, &src).unwrap();
        full.validate(wn, n).unwrap();
        let c1 = CompactCoeffs::load_in(&dev, &src, wn, n, true, Some(&dir)).unwrap();
        assert_eq!(c1.stats.source, CompactSource::Built);
        assert_eq!(c1.ntab, 15 + 13 - 5);
        let c2 = CompactCoeffs::load_in(&dev, &src, wn, n, true, Some(&dir)).unwrap();
        assert_eq!(c2.stats.source, CompactSource::Cache);
        assert!(c2.cs.is_mapped());
        let c3 = CompactCoeffs::load_in(&dev, &src, wn, n, false, Some(&dir)).unwrap();
        assert_eq!(c3.stats.source, CompactSource::Cache);
        assert!(!c3.cs.is_mapped());
        let c4 = CompactCoeffs::load_in(&dev, &src, wn, n, false, None).unwrap();
        assert_eq!(c4.stats.source, CompactSource::BuiltUncached);
        for c in [&c1, &c2, &c3, &c4] {
            let (a, b, faults) = both_outputs(&full, c, wn, n);
            assert_eq!(faults, [0, 0]);
            assert!(a.iter().any(|&x| x != 0));
            assert_eq!(a, b, "compact != full");
        }
        // The compact form is validated against the sizes.
        assert!(CompactCoeffs::load_in(&dev, &src, wn, 15, true, Some(&dir)).is_err());
        let max_s = full.coeffs.as_slice().iter().map(|c| c.s).max().unwrap() as usize;
        assert!(c2.validate(max_s, n).is_err());
        c2.validate(max_s + 1, n).unwrap();
        drop((c1, c2, c3, c4));

        // A damaged cache file is detected (checksum) and rebuilt; so is a truncated one.
        let files = cache_files(&dir);
        assert_eq!(files.len(), 1, "{files:?}");
        let mut cache = std::fs::read(&files[0]).unwrap();
        let at = ALIGN as usize + 5;
        cache[at] ^= 1;
        std::fs::write(&files[0], &cache).unwrap();
        let c5 = CompactCoeffs::load_in(&dev, &src, wn, n, true, Some(&dir)).unwrap();
        assert_eq!(c5.stats.source, CompactSource::Built);
        let cache = std::fs::read(&files[0]).unwrap();
        std::fs::write(&files[0], &cache[..cache.len() - 4]).unwrap();
        let c6 = CompactCoeffs::load_in(&dev, &src, wn, n, true, Some(&dir)).unwrap();
        assert_eq!(c6.stats.source, CompactSource::Built);
        // Another source file (different bytes) has another key.
        let mut other = bytes.clone();
        other[32 + 16] ^= 1;
        let src2 = temp_path("compact_src2.bin");
        std::fs::write(&src2, &other).unwrap();
        let c7 = CompactCoeffs::load_in(&dev, &src2, wn, n, true, Some(&dir)).unwrap();
        assert_eq!(c7.stats.source, CompactSource::Built);
        assert_eq!(cache_files(&dir).len(), 2);
        drop((c5, c6, c7));
        assert_eq!(invalidate_cache_in(&dir), 2);
        assert!(cache_files(&dir).is_empty());
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&src2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// R17 audit (ignored; run by tools/metal_audit/run_stress.sh's `groth16` family): the
    /// compact witness_into_poly on the real key, `R17_AUDIT_REPS` (default 10) times, against a
    /// CPU reference of a and b (arkworks), and the full-coefficient kernel once.
    #[test]
    #[ignore]
    fn r17_witness_poly_compact_repeat() {
        use ark_ff::{BigInteger as _, UniformRand as _};
        use rand::SeedableRng as _;

        let _lock = crate::metal::test_lock::exclusive();
        let Some(dir) = crate::metal::srs::tests::component_dir() else {
            return;
        };
        let reps: usize = std::env::var("R17_AUDIT_REPS").map_or(10, |v| v.parse().unwrap());
        let k = kernels().unwrap();
        let dev = &k.device;
        let path = dir.join("preprocessed_coeffs.bin");
        let hdr =
            crate::metal::srs::read_zkey_header(&dir.join("stark_verify_final.zkey")).unwrap();
        let (wn, n) = (hdr.witness_len(), hdr.domain_size);
        let full = Coeffs::load(dev, &path).unwrap();
        let cc = CompactCoeffs::load(dev, &path, wn, n, true).unwrap();
        // CPU reference (Montgomery limbs are ark's in-memory form).
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(17);
        let wit: Vec<Fr> = (0..wn).map(|_| Fr::rand(&mut rng)).collect();
        let mut want = vec![Fr::from(0u64); 2 * n];
        for (m, (co, ind)) in [
            (full.coeffs_a(), full.indices_a()),
            (full.coeffs_b(), full.indices_b()),
        ]
        .into_iter()
        .enumerate()
        {
            for r in 0..ind.len() - 1 {
                let mut sum = Fr::from(0u64);
                let mut row = 0;
                for c in &co[ind[r] as usize..ind[r + 1] as usize] {
                    sum += wit[c.s as usize] * Fr::new_unchecked(BigInt::new(c.value));
                    row = c.c as usize;
                }
                want[m * n + row] = sum;
            }
        }
        let want_bytes: Vec<u8> = want.iter().flat_map(|v| v.0.to_bytes_le()).collect();
        let wbytes: Vec<u8> = wit.iter().flat_map(|v| v.0.to_bytes_le()).collect();
        let shared = MTLResourceOptions::StorageModeShared;
        let w = dev.new_buffer_with_data(wbytes.as_ptr().cast(), wbytes.len() as u64, shared);
        let out = dev.new_buffer((2 * n * 32) as u64, shared);
        let fault = dev.new_buffer(16, shared);
        use crate::metal::prover::tests_support::{encode_witness_poly, CoeffsRef};
        let run = |c: CoeffsRef| -> (Vec<u8>, u32) {
            unsafe {
                std::ptr::write_bytes(out.contents() as *mut u8, 0, 2 * n * 32);
                std::ptr::write_bytes(fault.contents() as *mut u8, 0, 16);
            }
            let cb = k.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            encode_witness_poly(&k.small, enc, c, &out, n as u64, &w, wn as u64, &fault);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            unsafe {
                (
                    std::slice::from_raw_parts(out.contents() as *const u8, 2 * n * 32).to_vec(),
                    *(fault.contents() as *const u32),
                )
            }
        };
        let (full_out, f) = run(CoeffsRef::Full(&full));
        assert_eq!(f, 0);
        let full_bad = (full_out != want_bytes) as usize;
        let (first, f) = run(CoeffsRef::Compact(&cc));
        assert_eq!(f, 0);
        let (mut differ, mut wrong) = (0, (first != want_bytes) as usize);
        for _ in 1..reps {
            let (o, f) = run(CoeffsRef::Compact(&cc));
            assert_eq!(f, 0);
            differ += (o != first) as usize;
            wrong += (o != want_bytes) as usize;
        }
        eprintln!(
            "R17AUDIT groth16 witness_poly_compact: {differ}/{reps} reps differ from rep 0, \
             {wrong}/{reps} differ from CPU (full kernel: {full_bad}/1)"
        );
        assert_eq!((differ, wrong, full_bad), (0, 0, 0));
    }
}
