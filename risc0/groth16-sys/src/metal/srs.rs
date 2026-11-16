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

//! The proving key (`stark_verify_final.zkey` of the rzup `risc0-groth16` component) loaded into
//! shared Metal buffers (docs/metal-port/PLAN.md 4.5, item G3).
//!
//! Port of the CUDA prover's `SRS` constructor (`kernels/cuda/groth16_srs.cuh`) and the Intel
//! prover's `Srs` (`kernels/intel/groth16_intel.hpp`): the same snarkjs zkey layout and the same
//! point bytes. Points are stored as sppark's `affine_t`/`affine_fp2_t`: little-endian 4 x u64
//! limbs per base field element, in Montgomery form with R = 2^256 (the representation arkworks
//! keeps internally, so [G1AffineRaw::to_ark] is a reinterpretation, not a conversion), Fp2 as
//! (c0, c1), and the point at infinity as (0, 0).
//!
//! The loader reads (pread, several threads) straight into `StorageModeShared` buffers; nothing
//! of the point sections stays on the host besides the page cache. The buffers are read-only by
//! contract: kernels bind them as `const device`.

use std::{
    fs::File,
    marker::PhantomData,
    mem::size_of,
    os::unix::fs::FileExt,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, ensure, Context, Result};
use ark_bn254::{Fq, Fq2, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ff::{BigInt, PrimeField, Zero};
use metal::{
    foreign_types::{ForeignType, ForeignTypeRef},
    objc::{msg_send, runtime::Object, sel, sel_impl},
    Buffer, BufferRef, DeviceRef, MTLBuffer, MTLResourceOptions,
};

use super::filemap::FileMap;

// ---------------------------------------------------------------------------------------------
// Raw point types (the file and GPU layout)
// ---------------------------------------------------------------------------------------------

/// A base field element: 4 little-endian u64 limbs, Montgomery form (R = 2^256).
pub type FqRaw = [u64; 4];

/// sppark `affine_t` (64 bytes): x, y in Montgomery form; (0, 0) is the point at infinity.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct G1AffineRaw {
    pub x: FqRaw,
    pub y: FqRaw,
}

/// sppark `affine_fp2_t` (128 bytes): x = (x[0], x[1]) = x[0] + x[1] u, same for y.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct G2AffineRaw {
    pub x: [FqRaw; 2],
    pub y: [FqRaw; 2],
}

/// sppark `jacobian_t<fp_t>` (96 bytes): (X / Z^2, Y / Z^3); Z = 0 is the point at infinity.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct G1JacobianRaw {
    pub x: FqRaw,
    pub y: FqRaw,
    pub z: FqRaw,
}

/// sppark `jacobian_t<fp2_t>` (192 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct G2JacobianRaw {
    pub x: [FqRaw; 2],
    pub y: [FqRaw; 2],
    pub z: [FqRaw; 2],
}

const _: () = assert!(size_of::<G1AffineRaw>() == 64);
const _: () = assert!(size_of::<G2AffineRaw>() == 128);
const _: () = assert!(size_of::<G1JacobianRaw>() == 96);
const _: () = assert!(size_of::<G2JacobianRaw>() == 192);

/// Little-endian u64 limbs from 32 bytes.
pub(crate) fn limbs_le(b: &[u8]) -> [u64; 4] {
    assert_eq!(b.len(), 32);
    std::array::from_fn(|i| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap()))
}

/// The base field element whose Montgomery representation is `l`; errors if `l >= p`.
pub fn fq_from_mont(l: &FqRaw) -> Result<Fq> {
    let b = BigInt::new(*l);
    ensure!(
        b < Fq::MODULUS,
        "base field limbs {l:x?} are not below the modulus"
    );
    Ok(Fq::new_unchecked(b))
}

/// The scalar whose Montgomery representation is `l`; errors if `l >= r`.
pub fn fr_from_mont(l: &[u64; 4]) -> Result<Fr> {
    let b = BigInt::new(*l);
    ensure!(
        b < Fr::MODULUS,
        "scalar limbs {l:x?} are not below the modulus"
    );
    Ok(Fr::new_unchecked(b))
}

fn fq2_from_mont(l: &[FqRaw; 2]) -> Result<Fq2> {
    Ok(Fq2::new(fq_from_mont(&l[0])?, fq_from_mont(&l[1])?))
}

fn fq2_to_mont(v: &Fq2) -> [FqRaw; 2] {
    [v.c0.0 .0, v.c1.0 .0]
}

fn is_zero_limbs(l: &[FqRaw]) -> bool {
    l.iter().all(|e| e.iter().all(|&w| w == 0))
}

impl G1AffineRaw {
    pub fn is_infinity(&self) -> bool {
        is_zero_limbs(&[self.x, self.y])
    }

    /// The arkworks point (canonical limbs checked; curve membership is not: see
    /// `G1Affine::is_on_curve`).
    pub fn to_ark(&self) -> Result<G1Affine> {
        if self.is_infinity() {
            return Ok(G1Affine::identity());
        }
        Ok(G1Affine::new_unchecked(
            fq_from_mont(&self.x)?,
            fq_from_mont(&self.y)?,
        ))
    }

    pub fn from_ark(p: &G1Affine) -> Self {
        if p.infinity {
            return Self::default();
        }
        Self {
            x: p.x.0 .0,
            y: p.y.0 .0,
        }
    }
}

impl G2AffineRaw {
    pub fn is_infinity(&self) -> bool {
        is_zero_limbs(&[self.x[0], self.x[1], self.y[0], self.y[1]])
    }

    pub fn to_ark(&self) -> Result<G2Affine> {
        if self.is_infinity() {
            return Ok(G2Affine::identity());
        }
        Ok(G2Affine::new_unchecked(
            fq2_from_mont(&self.x)?,
            fq2_from_mont(&self.y)?,
        ))
    }

    pub fn from_ark(p: &G2Affine) -> Self {
        if p.infinity {
            return Self::default();
        }
        Self {
            x: fq2_to_mont(&p.x),
            y: fq2_to_mont(&p.y),
        }
    }
}

impl G1JacobianRaw {
    /// The arkworks point (arkworks' `Projective` is Jacobian too); Z = 0 maps to zero.
    pub fn to_ark(&self) -> Result<G1Projective> {
        let z = fq_from_mont(&self.z)?;
        if z.is_zero() {
            return Ok(G1Projective::zero());
        }
        Ok(G1Projective::new_unchecked(
            fq_from_mont(&self.x)?,
            fq_from_mont(&self.y)?,
            z,
        ))
    }
}

impl G2JacobianRaw {
    pub fn to_ark(&self) -> Result<G2Projective> {
        let z = fq2_from_mont(&self.z)?;
        if z.is_zero() {
            return Ok(G2Projective::zero());
        }
        Ok(G2Projective::new_unchecked(
            fq2_from_mont(&self.x)?,
            fq2_from_mont(&self.y)?,
            z,
        ))
    }
}

pub(crate) fn g1_affine_from_bytes(b: &[u8]) -> G1AffineRaw {
    G1AffineRaw {
        x: limbs_le(&b[0..32]),
        y: limbs_le(&b[32..64]),
    }
}

pub(crate) fn g2_affine_from_bytes(b: &[u8]) -> G2AffineRaw {
    G2AffineRaw {
        x: [limbs_le(&b[0..32]), limbs_le(&b[32..64])],
        y: [limbs_le(&b[64..96]), limbs_le(&b[96..128])],
    }
}

// ---------------------------------------------------------------------------------------------
// Shared buffers
// ---------------------------------------------------------------------------------------------

/// Plain-old-data element types a [SharedBuf] may hold: `repr(C)`, no padding bytes, every bit
/// pattern valid.
///
/// # Safety
/// Implementors must satisfy the above (the loaders fill the buffers with file bytes).
pub unsafe trait Pod: Copy + 'static {}
unsafe impl Pod for u32 {}
unsafe impl Pod for G1AffineRaw {}
unsafe impl Pod for G2AffineRaw {}

/// A `StorageModeShared` Metal buffer of `len` elements of `T`, filled once by a loader and
/// read-only afterwards (bind it as `const device T*` at [SharedBuf::offset]). Unlike zkp's
/// `BufferImpl` it is not pooled or hazard-tracked: nothing on the GPU writes it, so host reads
/// need no sync.
///
/// Either an allocated buffer the loader copied the file bytes into (offset 0), or (item G10) a
/// no-copy buffer over a read-only mapping of the file ([FileMap]; the elements start at the
/// section's offset inside its first page, so only 4-byte alignment is guaranteed).
pub struct SharedBuf<T: Pod> {
    backing: Backing,
    len: usize,
    _t: PhantomData<T>,
}

enum Backing {
    Owned(Buffer),
    Mapped(FileMap),
}

impl<T: Pod> SharedBuf<T> {
    /// A zero-length request still allocates 16 bytes (Metal rejects empty buffers).
    pub(crate) fn alloc(device: &DeviceRef, what: &str, len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| anyhow!("{what}: {len} elements overflow"))? as u64;
        let max = device.max_buffer_length();
        ensure!(
            bytes <= max,
            "{what}: {bytes} bytes exceed this GPU's max_buffer_length {max}"
        );
        let buffer = new_shared_buffer(device, bytes.max(16))
            .ok_or_else(|| anyhow!("Metal allocation of {bytes} bytes for {what} failed"))?;
        Ok(Self {
            backing: Backing::Owned(buffer),
            len,
            _t: PhantomData,
        })
    }

    /// `len` elements of `file` at byte `offset`, mapped without a copy ([FileMap::map]).
    pub(crate) fn map(
        device: &DeviceRef,
        file: &File,
        path: &Path,
        offset: u64,
        len: usize,
        what: &str,
    ) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| anyhow!("{what}: {len} elements overflow"))? as u64;
        ensure!(
            offset.is_multiple_of(4),
            "{what}: offset {offset} is not 4-byte aligned"
        );
        let map = FileMap::map(device, file, path, offset, bytes, what, true)?;
        Ok(Self {
            backing: Backing::Mapped(map),
            len,
            _t: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes of payload (`len * size_of::<T>()`; the allocation may be larger).
    pub fn byte_len(&self) -> u64 {
        (self.len * size_of::<T>()) as u64
    }

    /// Whether the elements are a no-copy mapping of the file.
    pub fn is_mapped(&self) -> bool {
        matches!(self.backing, Backing::Mapped(_))
    }

    /// The buffer to bind (as `const device T*`, at [Self::offset]).
    pub fn as_buf(&self) -> &BufferRef {
        match &self.backing {
            Backing::Owned(b) => b,
            Backing::Mapped(m) => m.buffer(),
        }
    }

    /// Byte offset of the first element in [Self::as_buf] (0 unless mapped).
    pub fn offset(&self) -> u64 {
        match &self.backing {
            Backing::Owned(_) => 0,
            Backing::Mapped(m) => m.inner_offset(),
        }
    }

    /// The payload bytes (host view; unified memory).
    pub fn bytes(&self) -> &[u8] {
        match &self.backing {
            // SAFETY: shared buffer of at least byte_len bytes, read-only by contract.
            Backing::Owned(b) => unsafe {
                std::slice::from_raw_parts(b.contents() as *const u8, self.byte_len() as usize)
            },
            Backing::Mapped(m) => m.bytes(),
        }
    }

    /// Element `i` (any alignment).
    pub fn get(&self, i: usize) -> T {
        assert!(i < self.len, "index {i} out of {}", self.len);
        // SAFETY: in bounds; T is Pod; read_unaligned has no alignment requirement.
        unsafe { (self.bytes().as_ptr() as *const T).add(i).read_unaligned() }
    }

    /// Host view of the contents (unified memory). Panics for a mapping whose offset is not
    /// aligned for `T` (use [Self::get] / [Self::bytes] there).
    pub fn as_slice(&self) -> &[T] {
        let p = self.bytes().as_ptr();
        assert!(
            (p as usize).is_multiple_of(std::mem::align_of::<T>()),
            "SharedBuf::as_slice on a mapping that is not aligned for the element type"
        );
        // SAFETY: at least len * size_of::<T>() bytes, aligned (checked), T is Pod (every bit
        // pattern valid), and GPU work only reads it (contract of this type), so no write races
        // this shared borrow.
        unsafe { std::slice::from_raw_parts(p as *const T, self.len) }
    }

    pub(crate) fn as_bytes_mut(&mut self) -> &mut [u8] {
        let n = self.len * size_of::<T>();
        match &self.backing {
            // SAFETY: as above; `&mut self` makes this the only view.
            Backing::Owned(b) => unsafe {
                std::slice::from_raw_parts_mut(b.contents() as *mut u8, n)
            },
            Backing::Mapped(_) => panic!("SharedBuf::as_bytes_mut on a read-only file mapping"),
        }
    }
}

/// `newBufferWithLength:options:` with a nil check on the raw pointer (metal-rs wraps it in a
/// NonNull, so a nil must be caught before; same approach as zkp's `Runtime::new_buffer_raw`).
// objc's `msg_send!` expands a `cfg(feature = "cargo-clippy")` this crate does not declare.
#[allow(unexpected_cfgs)]
fn new_shared_buffer(device: &DeviceRef, bytes: u64) -> Option<Buffer> {
    let options = MTLResourceOptions::StorageModeShared;
    // SAFETY: `newBufferWithLength:options:` returns a +1 (owned) reference or nil; only a
    // non-nil result is wrapped, and `from_ptr` takes over that reference.
    let ptr: *mut MTLBuffer = unsafe {
        msg_send![device.as_ptr() as *mut Object, newBufferWithLength: bytes options: options]
    };
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { Buffer::from_ptr(ptr) })
    }
}

/// Fill `dst` from `file` at `offset` with up to 8 threads issuing 16 MiB preads.
pub(crate) fn read_exact_at_par(file: &File, offset: u64, dst: &mut [u8]) -> std::io::Result<()> {
    const CHUNK: usize = 16 << 20;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 8);
    if dst.len() <= CHUNK || threads == 1 {
        return file.read_exact_at(dst, offset);
    }
    let per = dst.len().div_ceil(threads).div_ceil(CHUNK) * CHUNK;
    std::thread::scope(|s| {
        let handles: Vec<_> = dst
            .chunks_mut(per)
            .enumerate()
            .map(|(t, part)| {
                let base = offset + (t * per) as u64;
                s.spawn(move || -> std::io::Result<()> {
                    for (k, chunk) in part.chunks_mut(CHUNK).enumerate() {
                        file.read_exact_at(chunk, base + (k * CHUNK) as u64)?;
                    }
                    Ok(())
                })
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|h| h.join().expect("file reader thread panicked"))
    })
}

/// The Metal device's `currentAllocatedSize` (bytes of all live buffers of this process).
pub fn current_allocated_size(device: &DeviceRef) -> u64 {
    device.current_allocated_size()
}

// ---------------------------------------------------------------------------------------------
// zkey header
// ---------------------------------------------------------------------------------------------

/// One zkey section: payload offset and size in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Section {
    pub offset: u64,
    pub size: u64,
}

/// The verifying key as stored in zkey section 2 (sppark's `verifying_key`), plus the IC points
/// of section 3 (which the CUDA prover skips; arkworks' `gamma_abc_g1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyingKeyRaw {
    pub alpha_g1: G1AffineRaw,
    pub beta_g1: G1AffineRaw,
    pub beta_g2: G2AffineRaw,
    pub gamma_g2: G2AffineRaw,
    pub delta_g1: G1AffineRaw,
    pub delta_g2: G2AffineRaw,
    pub ic: Vec<G1AffineRaw>,
}

/// [VerifyingKeyRaw] as arkworks points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyingKeyArk {
    pub alpha_g1: G1Affine,
    pub beta_g1: G1Affine,
    pub beta_g2: G2Affine,
    pub gamma_g2: G2Affine,
    pub delta_g1: G1Affine,
    pub delta_g2: G2Affine,
    pub ic: Vec<G1Affine>,
}

impl VerifyingKeyRaw {
    pub fn to_ark(&self) -> Result<VerifyingKeyArk> {
        Ok(VerifyingKeyArk {
            alpha_g1: self.alpha_g1.to_ark()?,
            beta_g1: self.beta_g1.to_ark()?,
            beta_g2: self.beta_g2.to_ark()?,
            gamma_g2: self.gamma_g2.to_ark()?,
            delta_g1: self.delta_g1.to_ark()?,
            delta_g2: self.delta_g2.to_ark()?,
            ic: self
                .ic
                .iter()
                .map(G1AffineRaw::to_ark)
                .collect::<Result<_>>()?,
        })
    }
}

/// zkey section ids (snarkjs groth16).
pub mod section {
    pub const HEADER: usize = 1;
    pub const GROTH16_HEADER: usize = 2;
    pub const IC: usize = 3;
    pub const COEFFS: usize = 4;
    pub const POINTS_A: usize = 5;
    pub const POINTS_B1: usize = 6;
    pub const POINTS_B2: usize = 7;
    pub const POINTS_C: usize = 8;
    pub const POINTS_H: usize = 9;
}

/// Everything of a zkey except the point sections: read by [read_zkey_header] in a few small
/// preads, so callers can learn the sizes (the witness length) without loading the key.
#[derive(Clone, Debug)]
pub struct ZkeyHeader {
    pub file_size: u64,
    pub version: u32,
    /// Number of witness signals (= points in A, B1, B2).
    pub num_vars: usize,
    pub num_public: usize,
    /// Number of H points (a power of two).
    pub domain_size: usize,
    pub vk: VerifyingKeyRaw,
    /// Sections 1..=9 by id (index 0 unused).
    pub sections: [Section; 10],
    /// Number of (matrix, constraint, signal, value) records in section 4.
    pub num_coeffs: usize,
}

impl ZkeyHeader {
    /// Witness length the prover uploads: one scalar per A point (the CUDA `get_a_size`).
    pub fn witness_len(&self) -> usize {
        self.num_vars
    }

    /// Points of C: the private signals (`num_vars - num_public - 1`); the C MSM's scalars are the
    /// witness tail starting at `witness_len - c_len`.
    pub fn c_len(&self) -> usize {
        (self.sections[section::POINTS_C].size / 64) as usize
    }

    /// Payload bytes of the five point sections (what [Srs::load] puts in GPU buffers).
    pub fn points_bytes(&self) -> u64 {
        (section::POINTS_A..=section::POINTS_H)
            .map(|id| self.sections[id].size)
            .sum()
    }
}

/// BN254 base field and scalar field moduli, little-endian bytes (what zkey section 2 stores).
fn modulus_le_bytes<F: PrimeField<BigInt = BigInt<4>>>() -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, w) in F::MODULUS.0.iter().enumerate() {
        out[8 * i..8 * i + 8].copy_from_slice(&w.to_le_bytes());
    }
    out
}

struct Reader<'a> {
    file: &'a File,
    size: u64,
    path: &'a Path,
}

impl Reader<'_> {
    fn bytes(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let end = offset.checked_add(len as u64);
        if end.is_none_or(|e| e > self.size) {
            bail!(
                "proving key \"{}\" is truncated: needs bytes {offset}..{} of {}",
                self.path.display(),
                offset as u128 + len as u128,
                self.size
            );
        }
        let mut buf = vec![0u8; len];
        self.file
            .read_exact_at(&mut buf, offset)
            .with_context(|| format!("reading \"{}\"", self.path.display()))?;
        Ok(buf)
    }

    fn u32(&self, offset: u64) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.bytes(offset, 4)?.try_into().unwrap(),
        ))
    }

    fn u64(&self, offset: u64) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.bytes(offset, 8)?.try_into().unwrap(),
        ))
    }
}

fn read_header_from(file: &File, path: &Path) -> Result<ZkeyHeader> {
    let file_size = file
        .metadata()
        .with_context(|| format!("fstat(\"{}\")", path.display()))?
        .len();
    let r = Reader {
        file,
        size: file_size,
        path,
    };
    let p = path.display();
    ensure!(
        r.bytes(0, 4)? == b"zkey",
        "proving key \"{p}\" is not a zkey file"
    );
    let version = r.u32(4)?;
    let num_sections = r.u32(8)?;
    ensure!(
        num_sections == 10,
        "proving key \"{p}\" has {num_sections} sections, expected 10"
    );

    // Sections 1..=9 must come in ascending order (CUDA `check_section_id`); 10 is not read.
    let mut sections = [Section::default(); 10];
    let mut cursor = 12u64;
    for (id, sec) in sections.iter_mut().enumerate().skip(1) {
        let got = r.u32(cursor)?;
        ensure!(
            got as usize == id,
            "proving key \"{p}\": sections are expected in ascending order: expected section \
             {id}, got {got}"
        );
        let size = r.u64(cursor + 4)?;
        let offset = cursor + 12;
        ensure!(
            offset.checked_add(size).is_some_and(|e| e <= file_size),
            "proving key \"{p}\": section {id} ({size} bytes at {offset}) exceeds the file size \
             {file_size}"
        );
        *sec = Section { offset, size };
        cursor = offset + size;
    }

    // Section 1: protocol id.
    let s1 = sections[section::HEADER];
    ensure!(s1.size >= 4, "proving key \"{p}\": section 1 is too small");
    ensure!(
        r.u32(s1.offset)? == 1,
        "proving key \"{p}\": protocol is not groth16"
    );

    // Section 2: field moduli, sizes, verifying key (CUDA SRS(): 660 bytes for BN254).
    let s2 = sections[section::GROTH16_HEADER];
    let body = r.bytes(s2.offset, s2.size as usize)?;
    ensure!(
        body.len() == 4 + 32 + 4 + 32 + 12 + 3 * 64 + 3 * 128,
        "proving key \"{p}\": section 2 has {} bytes, expected 660 (BN254)",
        body.len()
    );
    let q_bytes = u32::from_le_bytes(body[0..4].try_into().unwrap());
    ensure!(
        q_bytes == 32,
        "proving key \"{p}\": base field is not 32 bytes"
    );
    ensure!(
        body[4..36] == modulus_le_bytes::<Fq>(),
        "proving key \"{p}\": base field modulus is not BN254's"
    );
    let r_bytes = u32::from_le_bytes(body[36..40].try_into().unwrap());
    ensure!(
        r_bytes == 32,
        "proving key \"{p}\": scalar field is not 32 bytes"
    );
    ensure!(
        body[40..72] == modulus_le_bytes::<Fr>(),
        "proving key \"{p}\": scalar field modulus is not BN254's"
    );
    let num_vars = u32::from_le_bytes(body[72..76].try_into().unwrap()) as usize;
    let num_public = u32::from_le_bytes(body[76..80].try_into().unwrap()) as usize;
    let domain_size = u32::from_le_bytes(body[80..84].try_into().unwrap()) as usize;
    let mut at = 84;
    let mut g1 = || {
        let v = g1_affine_from_bytes(&body[at..at + 64]);
        at += 64;
        v
    };
    let alpha_g1 = g1();
    let beta_g1 = g1();
    let mut at = 84 + 128;
    let mut g2 = || {
        let v = g2_affine_from_bytes(&body[at..at + 128]);
        at += 128;
        v
    };
    let beta_g2 = g2();
    let gamma_g2 = g2();
    let delta_g1 = g1_affine_from_bytes(&body[84 + 128 + 256..84 + 128 + 256 + 64]);
    let delta_g2 = g2_affine_from_bytes(&body[84 + 128 + 256 + 64..660]);

    // Section 3: IC (num_public + 1 G1 points).
    let s3 = sections[section::IC];
    ensure!(
        s3.size == (num_public as u64 + 1) * 64,
        "proving key \"{p}\": section 3 has {} bytes, expected {} IC points",
        s3.size,
        num_public + 1
    );
    let ic_bytes = r.bytes(s3.offset, s3.size as usize)?;
    let ic = ic_bytes
        .chunks_exact(64)
        .map(g1_affine_from_bytes)
        .collect();

    // Section 4: coefficient records (u32 count, then 44-byte records); not loaded here (the
    // preprocessed coefficients file is), only checked.
    let s4 = sections[section::COEFFS];
    let num_coeffs = r.u32(s4.offset)? as usize;
    ensure!(
        s4.size == 4 + num_coeffs as u64 * 44,
        "proving key \"{p}\": section 4 has {} bytes for {num_coeffs} coefficients",
        s4.size
    );

    ensure!(
        domain_size.is_power_of_two(),
        "proving key \"{p}\": size of h is expected to be a power of 2, but is {domain_size}"
    );

    // Points: A, B1, B2 have one point per signal, C one per private signal, H one per domain
    // element (a power of two).
    let count = |id: usize, elem: u64| -> Result<usize> {
        let s = sections[id];
        ensure!(
            s.size % elem == 0,
            "proving key \"{p}\": section {id} size {} is not a multiple of {elem}",
            s.size
        );
        Ok((s.size / elem) as usize)
    };
    for (id, elem, want) in [
        (section::POINTS_A, 64, num_vars),
        (section::POINTS_B1, 64, num_vars),
        (section::POINTS_B2, 128, num_vars),
        (
            section::POINTS_C,
            64,
            num_vars.checked_sub(num_public + 1).ok_or_else(|| {
                anyhow!("proving key \"{p}\": {num_vars} signals < {num_public} public + 1")
            })?,
        ),
        (section::POINTS_H, 64, domain_size),
    ] {
        let n = count(id, elem)?;
        ensure!(
            n == want,
            "proving key \"{p}\": section {id} has {n} points, expected {want}"
        );
    }

    Ok(ZkeyHeader {
        file_size,
        version,
        num_vars,
        num_public,
        domain_size,
        vk: VerifyingKeyRaw {
            alpha_g1,
            beta_g1,
            beta_g2,
            gamma_g2,
            delta_g1,
            delta_g2,
            ic,
        },
        sections,
        num_coeffs,
    })
}

/// Parse and check the zkey header, the verifying key and the section table (a few KB read).
pub fn read_zkey_header(path: &Path) -> Result<ZkeyHeader> {
    let file = File::open(path).with_context(|| format!("open(\"{}\")", path.display()))?;
    read_header_from(&file, path)
}

// ---------------------------------------------------------------------------------------------
// SRS
// ---------------------------------------------------------------------------------------------

/// Timings and memory of one load.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoadStats {
    /// Wall time of the whole load (open, header, allocation, reads).
    pub elapsed: Duration,
    /// Payload bytes copied into GPU buffers.
    pub bytes: u64,
    /// Device `currentAllocatedSize` before and after the load (process-wide).
    pub allocated_before: u64,
    pub allocated_after: u64,
}

impl LoadStats {
    /// Read throughput in GB/s.
    pub fn gb_per_s(&self) -> f64 {
        self.bytes as f64 / self.elapsed.as_secs_f64().max(1e-9) / 1e9
    }
}

/// The proving key in shared Metal buffers: the CUDA `SRS` (`a`, `b_g1`, `b_g2`, `c`, `h` and the
/// verifying key), plus the IC points and the header.
pub struct Srs {
    pub header: ZkeyHeader,
    pub a: SharedBuf<G1AffineRaw>,
    pub b_g1: SharedBuf<G1AffineRaw>,
    pub b_g2: SharedBuf<G2AffineRaw>,
    pub c: SharedBuf<G1AffineRaw>,
    pub h: SharedBuf<G1AffineRaw>,
    pub stats: LoadStats,
}

impl Srs {
    /// Load `stark_verify_final.zkey` (`SetupParams::srs_path`) into allocated buffers. Takes no
    /// lock and needs no `MetalHal` (G1: `metal_preload` may run beside a STARK proof).
    pub fn load(device: &DeviceRef, path: &Path) -> Result<Self> {
        Self::load_with(device, path, false)
    }

    /// [Self::load], or with `mapped` (item G10) the point sections as no-copy buffers over a
    /// read-only mapping of the file ([SharedBuf::map]): nothing is copied, and the points are
    /// the file's page cache instead of 2.3 GB of anonymous memory.
    pub fn load_with(device: &DeviceRef, path: &Path, mapped: bool) -> Result<Self> {
        let start = Instant::now();
        let allocated_before = device.current_allocated_size();
        let file = File::open(path).with_context(|| format!("open(\"{}\")", path.display()))?;
        let header = read_header_from(&file, path)?;

        let load_section = |sec: Section, what: &str| -> Result<SharedBuf<G1AffineRaw>> {
            load_section_as(device, &file, path, sec, what, mapped)
        };
        fn load_section_as<T: Pod>(
            device: &DeviceRef,
            file: &File,
            path: &Path,
            sec: Section,
            what: &str,
            mapped: bool,
        ) -> Result<SharedBuf<T>> {
            let len = sec.size as usize / size_of::<T>();
            if mapped {
                return SharedBuf::<T>::map(device, file, path, sec.offset, len, what);
            }
            let mut buf = SharedBuf::<T>::alloc(device, what, len)?;
            read_exact_at_par(file, sec.offset, buf.as_bytes_mut())
                .with_context(|| format!("reading {what} from \"{}\"", path.display()))?;
            Ok(buf)
        }
        let s = &header.sections;
        let a = load_section(s[section::POINTS_A], "SRS points A")?;
        let b_g1 = load_section(s[section::POINTS_B1], "SRS points B1")?;
        let b_g2 = load_section_as(
            device,
            &file,
            path,
            s[section::POINTS_B2],
            "SRS points B2",
            mapped,
        )?;
        let c = load_section(s[section::POINTS_C], "SRS points C")?;
        let h = load_section(s[section::POINTS_H], "SRS points H")?;
        let stats = LoadStats {
            elapsed: start.elapsed(),
            bytes: header.points_bytes(),
            allocated_before,
            allocated_after: device.current_allocated_size(),
        };
        Ok(Self {
            header,
            a,
            b_g1,
            b_g2,
            c,
            h,
            stats,
        })
    }

    pub fn vk(&self) -> &VerifyingKeyRaw {
        &self.header.vk
    }

    /// Payload bytes held in GPU buffers.
    pub fn resident_bytes(&self) -> u64 {
        self.a.byte_len()
            + self.b_g1.byte_len()
            + self.b_g2.byte_len()
            + self.c.byte_len()
            + self.h.byte_len()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::PathBuf;

    use ark_bn254::Bn254;
    use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
    use ark_ff::{Field, UniformRand};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    /// The rzup `risc0-groth16` component version whose proving key matches the 3.1 verifier's
    /// key (the v0.2.0 ceremony); risc0-groth16's `prove::component` resolves the same one.
    pub(crate) const COMPONENT_VERSION: &str = "0.2.0";

    /// The rzup component directory, if its files are present: `RISC0_GROTH16_DIR`, else rzup's
    /// layout `$RISC0_HOME` (else `~/.risc0`) `/extensions/v0.2.0-risc0-groth16`, as
    /// risc0-groth16's resolver falls back to (this crate does not depend on rzup). Never an
    /// older component. Tests that need it print why they skip.
    pub(crate) fn component_dir() -> Option<PathBuf> {
        let dir = match std::env::var_os("RISC0_GROTH16_DIR") {
            Some(dir) => PathBuf::from(dir),
            None => {
                let home = std::env::var_os("RISC0_HOME")
                    .map(PathBuf::from)
                    .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".risc0")));
                let Some(home) = home else {
                    eprintln!("skipped: no home directory to find the risc0-groth16 component in");
                    return None;
                };
                home.join("extensions")
                    .join(format!("v{COMPONENT_VERSION}-risc0-groth16"))
            }
        };
        if dir.join("stark_verify_final.zkey").is_file() {
            Some(dir)
        } else {
            eprintln!(
                "skipped: no risc0-groth16 component at {} (rzup install risc0-groth16 \
                 {COMPONENT_VERSION}, or set RISC0_GROTH16_DIR)",
                dir.display()
            );
            None
        }
    }

    /// This process's resident set size in bytes (`ps -o rss=`), for the load reports.
    pub(crate) fn rss_bytes() -> u64 {
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .expect("ps");
        String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u64>()
            .unwrap_or(0)
            * 1024
    }

    pub(crate) fn device() -> metal::Device {
        metal::Device::system_default().expect("no Metal device")
    }

    /// A unique temporary path (removed by the caller).
    pub(crate) fn temp_path(name: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("risc0-g16-{}-{n}-{name}", std::process::id()))
    }

    fn g1_bytes(p: &G1Affine) -> Vec<u8> {
        let r = G1AffineRaw::from_ark(p);
        r.x.iter()
            .chain(&r.y)
            .flat_map(|w| w.to_le_bytes())
            .collect()
    }

    fn g2_bytes(p: &G2Affine) -> Vec<u8> {
        let r = G2AffineRaw::from_ark(p);
        r.x.iter()
            .chain(&r.y)
            .flatten()
            .flat_map(|w| w.to_le_bytes())
            .collect()
    }

    /// A small synthetic zkey in the snarkjs layout with random points (and some infinities).
    pub(crate) struct Synthetic {
        pub bytes: Vec<u8>,
        pub vk: VerifyingKeyArk,
        pub a: Vec<G1Affine>,
        pub b_g1: Vec<G1Affine>,
        pub b_g2: Vec<G2Affine>,
        pub c: Vec<G1Affine>,
        pub h: Vec<G1Affine>,
    }

    pub(crate) fn synthetic(num_vars: usize, num_public: usize, domain: usize) -> Synthetic {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut g1 = |i: usize| {
            if i % 7 == 3 {
                G1Affine::identity()
            } else {
                (G1Affine::generator() * Fr::rand(&mut rng)).into_affine()
            }
        };
        let vk_g1: Vec<_> = (0..3).map(|i| g1(i + 1)).collect();
        let ic: Vec<_> = (0..=num_public).map(|i| g1(i + 1)).collect();
        let a: Vec<_> = (0..num_vars).map(&mut g1).collect();
        let b_g1: Vec<_> = (0..num_vars).map(&mut g1).collect();
        let c: Vec<_> = (0..num_vars - num_public - 1).map(&mut g1).collect();
        let h: Vec<_> = (0..domain).map(&mut g1).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(8);
        let mut g2 = |i: usize| {
            if i % 5 == 2 {
                G2Affine::identity()
            } else {
                (G2Affine::generator() * Fr::rand(&mut rng)).into_affine()
            }
        };
        let vk_g2: Vec<_> = (0..3).map(|i| g2(i + 3)).collect();
        let b_g2: Vec<_> = (0..num_vars).map(&mut g2).collect();

        let mut sec: Vec<Vec<u8>> = vec![vec![]; 11];
        sec[1] = 1u32.to_le_bytes().to_vec();
        let mut s2 = Vec::new();
        s2.extend(32u32.to_le_bytes());
        s2.extend(modulus_le_bytes::<Fq>());
        s2.extend(32u32.to_le_bytes());
        s2.extend(modulus_le_bytes::<Fr>());
        s2.extend((num_vars as u32).to_le_bytes());
        s2.extend((num_public as u32).to_le_bytes());
        s2.extend((domain as u32).to_le_bytes());
        s2.extend(g1_bytes(&vk_g1[0]));
        s2.extend(g1_bytes(&vk_g1[1]));
        s2.extend(g2_bytes(&vk_g2[0]));
        s2.extend(g2_bytes(&vk_g2[1]));
        s2.extend(g1_bytes(&vk_g1[2]));
        s2.extend(g2_bytes(&vk_g2[2]));
        sec[2] = s2;
        sec[3] = ic.iter().flat_map(g1_bytes).collect();
        let mut s4 = 2u32.to_le_bytes().to_vec();
        s4.extend([0u8; 88]);
        sec[4] = s4;
        sec[5] = a.iter().flat_map(g1_bytes).collect();
        sec[6] = b_g1.iter().flat_map(g1_bytes).collect();
        sec[7] = b_g2.iter().flat_map(g2_bytes).collect();
        sec[8] = c.iter().flat_map(g1_bytes).collect();
        sec[9] = h.iter().flat_map(g1_bytes).collect();
        sec[10] = vec![0u8; 5];

        let mut bytes = b"zkey".to_vec();
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(10u32.to_le_bytes());
        for (id, s) in sec.iter().enumerate().skip(1) {
            bytes.extend((id as u32).to_le_bytes());
            bytes.extend((s.len() as u64).to_le_bytes());
            bytes.extend(s);
        }
        Synthetic {
            bytes,
            vk: VerifyingKeyArk {
                alpha_g1: vk_g1[0],
                beta_g1: vk_g1[1],
                beta_g2: vk_g2[0],
                gamma_g2: vk_g2[1],
                delta_g1: vk_g1[2],
                delta_g2: vk_g2[2],
                ic,
            },
            a,
            b_g1,
            b_g2,
            c,
            h,
        }
    }

    fn to_ark_g1(v: &[G1AffineRaw]) -> Vec<G1Affine> {
        v.iter().map(|p| p.to_ark().unwrap()).collect()
    }

    #[test]
    fn raw_round_trip_and_canonical_check() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for _ in 0..64 {
            let p = (G1Affine::generator() * Fr::rand(&mut rng)).into_affine();
            assert_eq!(G1AffineRaw::from_ark(&p).to_ark().unwrap(), p);
            let q = (G2Affine::generator() * Fr::rand(&mut rng)).into_affine();
            assert_eq!(G2AffineRaw::from_ark(&q).to_ark().unwrap(), q);
        }
        assert!(G1AffineRaw::default().to_ark().unwrap().is_zero());
        assert!(G2AffineRaw::default().to_ark().unwrap().is_zero());
        // Montgomery: the limbs of 1 are R mod p, not 1.
        assert_ne!(Fq::from(1u64).0 .0, [1, 0, 0, 0]);
        assert_eq!(fq_from_mont(&Fq::from(1u64).0 .0).unwrap(), Fq::from(1u64));
        // R^-1 has limbs [1, 0, 0, 0]: x * R^-1 * R = x.
        let r_inv = fq_from_mont(&[1, 0, 0, 0]).unwrap();
        assert_eq!(r_inv * Fq::from(2u64).pow([256]), Fq::from(1u64));
        assert!(fq_from_mont(&Fq::MODULUS.0).is_err());
        assert!(fr_from_mont(&Fr::MODULUS.0).is_err());
        // Jacobian with Z = 0 is zero; Z = 1 is the affine point.
        let g = G1Affine::generator();
        let one = Fq::from(1u64).0 .0;
        let j = G1JacobianRaw {
            x: g.x.0 .0,
            y: g.y.0 .0,
            z: one,
        };
        assert_eq!(j.to_ark().unwrap().into_affine(), g);
        assert!(G1JacobianRaw { z: [0; 4], ..j }.to_ark().unwrap().is_zero());
    }

    #[test]
    fn synthetic_zkey_loads() {
        let syn = synthetic(40, 3, 16);
        let path = temp_path("syn.zkey");
        std::fs::write(&path, &syn.bytes).unwrap();
        let dev = device();
        let srs = Srs::load(&dev, &path);
        std::fs::remove_file(&path).unwrap();
        let srs = srs.unwrap();
        assert_eq!(srs.header.num_vars, 40);
        assert_eq!(srs.header.num_public, 3);
        assert_eq!(srs.header.domain_size, 16);
        assert_eq!(srs.header.num_coeffs, 2);
        assert_eq!(srs.header.c_len(), 36);
        assert_eq!(srs.vk().to_ark().unwrap(), syn.vk);
        assert_eq!(to_ark_g1(srs.a.as_slice()), syn.a);
        assert_eq!(to_ark_g1(srs.b_g1.as_slice()), syn.b_g1);
        let b2: Vec<_> = srs
            .b_g2
            .as_slice()
            .iter()
            .map(|p| p.to_ark().unwrap())
            .collect();
        assert_eq!(b2, syn.b_g2);
        assert_eq!(to_ark_g1(srs.c.as_slice()), syn.c);
        assert_eq!(to_ark_g1(srs.h.as_slice()), syn.h);
        assert_eq!(
            srs.resident_bytes(),
            (40 * 64 * 2 + 40 * 128 + 36 * 64 + 16 * 64) as u64
        );
        assert_eq!(srs.stats.bytes, srs.resident_bytes());
    }

    #[test]
    fn malformed_zkeys_are_rejected() {
        let syn = synthetic(12, 1, 4);
        let dev = device();
        // (mutation, expected error fragment)
        let s2 = 12 + 12 + 4 + 12; // payload offset of section 2
        let cases: Vec<(Box<dyn Fn(&mut Vec<u8>)>, &str)> = vec![
            (Box::new(|b| b[0] = b'Z'), "not a zkey file"),
            (Box::new(|b| b[8] = 9), "has 9 sections"),
            (Box::new(|b| b[24 + 4] = 3), "ascending order"),
            (
                Box::new(|b| b.truncate(b.len() - 200)),
                "exceeds the file size",
            ),
            (Box::new(|b| b[24] = 2), "protocol is not groth16"),
            (Box::new(move |b| b[s2 + 4] ^= 1), "base field modulus"),
            (Box::new(move |b| b[s2 + 40] ^= 1), "scalar field modulus"),
            (Box::new(move |b| b[s2 + 80] = 5), "power of 2"),
            (Box::new(move |b| b[s2 + 72] = 13), "expected 13"),
        ];
        for (i, (mutate, want)) in cases.iter().enumerate() {
            let mut bytes = syn.bytes.clone();
            mutate(&mut bytes);
            let path = temp_path("bad.zkey");
            std::fs::write(&path, &bytes).unwrap();
            let res = Srs::load(&dev, &path);
            std::fs::remove_file(&path).unwrap();
            let err = format!(
                "{:#}",
                res.err().unwrap_or_else(|| panic!("case {i} loaded"))
            );
            assert!(err.contains(want), "case {i}: {err:?} lacks {want:?}");
        }
        assert!(Srs::load(&dev, Path::new("/nonexistent/x.zkey")).is_err());
    }

    /// Indices sampled across `len`: the ends plus `n` evenly spaced.
    pub(crate) fn sample(len: usize, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).map(|k| k * len / n).collect();
        v.extend([0, 1, len.saturating_sub(2), len.saturating_sub(1)]);
        v.retain(|&i| i < len);
        v.sort();
        v.dedup();
        v
    }

    /// The real proving key: header consistency, VK pairing relations, spot-checked points on the
    /// curve and in the subgroup, load time and memory.
    #[test]
    fn component_srs() {
        // Loads the real key (GBs): alone (PLAN 8.7).
        let _lock = super::super::test_lock::exclusive();
        let Some(dir) = component_dir() else { return };
        let path = dir.join("stark_verify_final.zkey");
        let dev = device();
        let rss0 = rss_bytes();
        let srs = Srs::load(&dev, &path).unwrap();
        println!(
            "process RSS {:.3} -> {:.3} GB",
            rss0 as f64 / 1e9,
            rss_bytes() as f64 / 1e9
        );
        let h = &srs.header;
        let st = srs.stats;
        println!(
            "zkey: {} vars, {} public, domain 2^{}, {} coeffs; points {:.3} GB loaded in {:.3} s \
             ({:.2} GB/s); device allocated {:.3} -> {:.3} GB",
            h.num_vars,
            h.num_public,
            h.domain_size.trailing_zeros(),
            h.num_coeffs,
            srs.resident_bytes() as f64 / 1e9,
            st.elapsed.as_secs_f64(),
            st.gb_per_s(),
            st.allocated_before as f64 / 1e9,
            st.allocated_after as f64 / 1e9,
        );
        // current_allocated_size is the device total, shared with tests running concurrently in
        // this process (MSM, NTT, small kernels), which can free memory during the load; so
        // only the total, which must include the live SRS buffers, is checked, not the delta.
        assert!(st.allocated_after >= srs.resident_bytes());

        // The VK: points valid, and beta_g1/delta_g1 match beta_g2/delta_g2 (e(x G1, G2) =
        // e(G1, x G2)), which ties the G1 and G2 decodings (including Fp2 order) together.
        let vk = srs.vk().to_ark().unwrap();
        for p in [vk.alpha_g1, vk.beta_g1, vk.delta_g1].iter().chain(&vk.ic) {
            assert!(
                !p.is_zero() && p.is_on_curve() && p.is_in_correct_subgroup_assuming_on_curve()
            );
        }
        for p in [vk.beta_g2, vk.gamma_g2, vk.delta_g2] {
            assert!(
                !p.is_zero() && p.is_on_curve() && p.is_in_correct_subgroup_assuming_on_curve()
            );
        }
        let (g1, g2) = (G1Affine::generator(), G2Affine::generator());
        assert_eq!(
            Bn254::pairing(vk.beta_g1, g2),
            Bn254::pairing(g1, vk.beta_g2)
        );
        assert_eq!(
            Bn254::pairing(vk.delta_g1, g2),
            Bn254::pairing(g1, vk.delta_g2)
        );
        assert_eq!(vk.ic.len(), h.num_public + 1);

        // Spot checks: every sampled point decodes (canonical limbs) and is on the curve and in
        // the subgroup, or is the point at infinity.
        let mut infinities = 0;
        let mut checked = 0;
        for (name, pts) in [
            ("A", srs.a.as_slice()),
            ("B1", srs.b_g1.as_slice()),
            ("C", srs.c.as_slice()),
            ("H", srs.h.as_slice()),
        ] {
            for i in sample(pts.len(), 2048) {
                let p = pts[i]
                    .to_ark()
                    .unwrap_or_else(|e| panic!("{name}[{i}]: {e}"));
                infinities += p.is_zero() as usize;
                assert!(p.is_on_curve(), "{name}[{i}] not on the curve");
                assert!(p.is_in_correct_subgroup_assuming_on_curve(), "{name}[{i}]");
                checked += 1;
            }
        }
        for i in sample(srs.b_g2.len(), 2048) {
            let p = srs.b_g2.as_slice()[i].to_ark().unwrap();
            infinities += p.is_zero() as usize;
            assert!(
                p.is_on_curve() && p.is_in_correct_subgroup_assuming_on_curve(),
                "B2[{i}]"
            );
            checked += 1;
        }
        println!("spot-checked {checked} points ({infinities} at infinity)");
        assert!(infinities < checked);
    }
}
