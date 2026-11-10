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

//! Host wrappers for the small Groth16 kernels (item G4, docs/metal-port/PLAN.md 4.5):
//! `kernels/metal/{eltwise,witness_poly,fuzz,powers}.metal`.
//!
//! | kernel | replaces (CUDA prover) | wrapper |
//! |---|---|---|
//! | `fr_mul` | `coeff_wise_mul` | [SmallKernels::fr_mul] |
//! | `fr_mul_sub_from_mont` | `coeff_wise_mul_and_sub` + the H scalars' `from_mont` | [SmallKernels::fr_mul_sub_from_mont] |
//! | `fr_add` | `coeff_wise_add` (witness + fuzz) | [SmallKernels::fr_add] |
//! | `witness_into_poly` | `witness_into_poly` | [SmallKernels::witness_into_poly] |
//! | `chacha_fuzz` | `chacha_generate_random_scalars<8>` | [SmallKernels::chacha_fuzz] |
//! | `fr_powers`, `fr_scale` | `generate_partial_twiddles` / the Intel `powers` + coset scale | [SmallKernels::fr_powers], [SmallKernels::fr_scale] |
//!
//! The wrappers only **encode** into a caller-provided compute encoder (one dispatch each, no
//! waits), so the prover (G7) decides the command-buffer boundaries. Buffers are addressed as
//! [BufSlice] (buffer + byte offset); field elements are `bn254::Fr` on the GPU: 8 little-endian
//! u32 limbs in Montgomery form with R = 2^256, which is ark-ff's in-memory representation of
//! [ark_bn254::Fr] (see [fr_to_words] / [fr_from_words]). Lengths and offsets are checked against
//! the buffers (a violation is a programming error and panics).
//!
//! Hazards: every buffer is a tracked `StorageModeShared` buffer and a compute encoder created
//! with the default (serial) dispatch type orders dependent dispatches, so consecutive wrappers
//! on one encoder see each other's writes.

use std::ffi::c_void;

use anyhow::{anyhow, Result};
use ark_bn254::Fr;
use ark_ff::{BigInt, Field};
use metal::{
    BufferRef, ComputeCommandEncoderRef, ComputePipelineState, DeviceRef, LibraryRef, MTLSize,
};

/// Bytes of one `bn254::Fr` on the GPU.
pub const FR_BYTES: u64 = 32;
/// Bytes of one preprocessed coefficient (`Coeff { m, c, s, pad: u32, value: Fr }`, sppark's
/// `coeff_t`).
pub const COEFF_BYTES: u64 = 48;
/// The threadgroup width every small kernel is dispatched with (`max_total_threads_per_threadgroup`
/// in the sources); the PSO's own limit wins if smaller.
const TG: u64 = 256;
/// The CUDA warp size the fuzz layout is defined with (not the Metal SIMD width).
const FUZZ_WARP: u64 = 32;

/// The kernel entry points of this module, in `METAL_KERNEL_NAMES`.
pub const SMALL_KERNEL_NAMES: &[&str] = &[
    "fr_mul",
    "fr_mul_sub_from_mont",
    "fr_add",
    "witness_into_poly",
    "chacha_fuzz",
    "fr_powers",
    "fr_scale",
];

/// One preprocessed coefficient as the kernels read it (`kernels/metal/witness_poly.metal`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Coeff {
    /// Matrix: 0 = A, 1 = B.
    pub m: u32,
    /// Output row (constraint).
    pub c: u32,
    /// Witness index (signal).
    pub s: u32,
    pub pad: u32,
    /// The coefficient (raw limbs as stored in the file).
    pub value: [u32; 8],
}
const _: () = assert!(std::mem::size_of::<Coeff>() == COEFF_BYTES as usize);

/// `fault` bits of [SmallKernels::witness_into_poly].
pub mod witness_fault {
    /// Row indices not monotone or past the coefficient array (row skipped).
    pub const BAD_RANGE: u32 = 1;
    /// A coefficient's `s` or `c` out of range (row skipped).
    pub const BAD_INDEX: u32 = 2;
    /// A non-canonical coefficient value or witness element (row still computed).
    pub const NON_CANONICAL: u32 = 4;
}

/// A region of a Metal buffer starting `offset` bytes in.
#[derive(Clone, Copy)]
pub struct BufSlice<'a> {
    pub buf: &'a BufferRef,
    pub offset: u64,
}

impl<'a> BufSlice<'a> {
    /// The whole buffer.
    pub fn new(buf: &'a BufferRef) -> Self {
        Self { buf, offset: 0 }
    }

    /// `buf` from byte `offset`.
    pub fn at(buf: &'a BufferRef, offset: u64) -> Self {
        Self { buf, offset }
    }

    /// `buf` from Fr element `index` (`index * 32` bytes).
    pub fn fr_at(buf: &'a BufferRef, index: u64) -> Self {
        Self::at(buf, index * FR_BYTES)
    }

    /// Panics unless `bytes` bytes from the offset lie inside the buffer.
    fn check(&self, bytes: u64, what: &str) {
        let end = self.offset.checked_add(bytes);
        assert!(
            end.is_some_and(|e| e <= self.buf.length()),
            "{what}: {bytes} bytes at offset {} exceed the buffer ({} bytes)",
            self.offset,
            self.buf.length()
        );
    }
}

/// Montgomery limbs of an ark Fr (its in-memory form) as the GPU's 8 x u32.
pub fn fr_to_words(v: &Fr) -> [u32; 8] {
    let mut w = [0u32; 8];
    for (i, l) in v.0 .0.iter().enumerate() {
        w[2 * i] = *l as u32;
        w[2 * i + 1] = (*l >> 32) as u32;
    }
    w
}

/// The ark Fr whose Montgomery limbs are `w` (no reduction: `w` must be < the modulus for ark
/// arithmetic to be meaningful).
pub fn fr_from_words(w: &[u32]) -> Fr {
    let mut l = [0u64; 4];
    for i in 0..4 {
        l[i] = w[2 * i] as u64 | ((w[2 * i + 1] as u64) << 32);
    }
    Fr::new_unchecked(BigInt::new(l))
}

/// Number of ChaCha blocks for `n` fuzz scalars (`ceil(n * 32 / 64)`, the CUDA prover's
/// `generate_random_scalars`); the fuzz buffer holds `2 * blocks` Fr.
pub fn fuzz_blocks(n: u64) -> u64 {
    (n * FR_BYTES).div_ceil(64)
}

/// The initial ChaCha state of the CUDA prover's `initialize_chacha_state<uint32_t>(key, nonce)`:
/// constants, key words, counter 0 in word 12, the 12-byte nonce in words 13..15 (little-endian).
pub fn chacha_state(key: &[u8; 32], nonce: &[u8; 12]) -> [u32; 16] {
    let word = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let mut s = [0u32; 16];
    s[..4].copy_from_slice(&[0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]);
    for i in 0..8 {
        s[4 + i] = word(&key[4 * i..]);
    }
    s[12] = 0;
    for i in 0..3 {
        s[13 + i] = word(&nonce[4 * i..]);
    }
    s
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FrEltParams {
    count: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WitnessPolyParams {
    rows: u32,
    coeff_len: u32,
    witness_len: u32,
    out_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ChachaFuzzParams {
    state: [u32; 16],
    blocks: u32,
    threads: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FrPowersParams {
    sq: [[u32; 8]; 32],
    scale: [u32; 8],
    count: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FrScaleParams {
    scale: [u32; 8],
    count: u32,
    _pad: [u32; 3],
}

// Sizes of the MSL structs (all members 4-byte, no implicit padding).
const _: () = assert!(std::mem::size_of::<FrEltParams>() == 16);
const _: () = assert!(std::mem::size_of::<WitnessPolyParams>() == 16);
const _: () = assert!(std::mem::size_of::<ChachaFuzzParams>() == 80);
const _: () = assert!(std::mem::size_of::<FrPowersParams>() == 1072);
const _: () = assert!(std::mem::size_of::<FrScaleParams>() == 48);

fn count_u32(n: u64, what: &str) -> u32 {
    u32::try_from(n).unwrap_or_else(|_| panic!("{what}: {n} elements exceed u32"))
}

/// The pipelines of the small kernels, built once from the groth16 metallib.
pub struct SmallKernels {
    fr_mul: ComputePipelineState,
    fr_mul_sub_from_mont: ComputePipelineState,
    fr_add: ComputePipelineState,
    witness_into_poly: ComputePipelineState,
    chacha_fuzz: ComputePipelineState,
    fr_powers: ComputePipelineState,
    fr_scale: ComputePipelineState,
}

impl SmallKernels {
    /// Build the PSOs from `library` (normally `device.new_library_with_data(METAL_LIB)`).
    pub fn new(device: &DeviceRef, library: &LibraryRef) -> Result<Self> {
        let pso = |name: &str| -> Result<ComputePipelineState> {
            let f = library
                .get_function(name, None)
                .map_err(|e| anyhow!("groth16 metallib: kernel {name}: {e}"))?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| anyhow!("groth16 metallib: pipeline {name}: {e}"))
        };
        Ok(Self {
            fr_mul: pso("fr_mul")?,
            fr_mul_sub_from_mont: pso("fr_mul_sub_from_mont")?,
            fr_add: pso("fr_add")?,
            witness_into_poly: pso("witness_into_poly")?,
            chacha_fuzz: pso("chacha_fuzz")?,
            fr_powers: pso("fr_powers")?,
            fr_scale: pso("fr_scale")?,
        })
    }

    fn dispatch<P: Copy>(
        enc: &ComputeCommandEncoderRef,
        pso: &ComputePipelineState,
        bufs: &[BufSlice],
        params: &P,
        threads: u64,
    ) {
        if threads == 0 {
            return;
        }
        enc.set_compute_pipeline_state(pso);
        for (i, b) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b.buf), b.offset);
        }
        enc.set_bytes(
            bufs.len() as u64,
            std::mem::size_of::<P>() as u64,
            params as *const P as *const c_void,
        );
        let tg = pso.max_total_threads_per_threadgroup().min(TG);
        enc.dispatch_threads(MTLSize::new(threads, 1, 1), MTLSize::new(tg, 1, 1));
    }

    /// `out[i] = a[i] * b[i]` for `i < n` (`out` may be `a` or `b`).
    pub fn fr_mul(
        &self,
        enc: &ComputeCommandEncoderRef,
        out: BufSlice,
        a: BufSlice,
        b: BufSlice,
        n: u64,
    ) {
        for (s, w) in [(&out, "fr_mul out"), (&a, "fr_mul a"), (&b, "fr_mul b")] {
            s.check(n * FR_BYTES, w);
        }
        let p = FrEltParams {
            count: count_u32(n, "fr_mul"),
            _pad: [0; 3],
        };
        Self::dispatch(enc, &self.fr_mul, &[out, a, b], &p, n);
    }

    /// `inout[i] = (inout[i] * b[i] - c[i]).from_mont()` for `i < n`: the H scalars of the prover
    /// (a * b - c over the coset), converted to canonical integers for the MSM.
    pub fn fr_mul_sub_from_mont(
        &self,
        enc: &ComputeCommandEncoderRef,
        inout: BufSlice,
        b: BufSlice,
        c: BufSlice,
        n: u64,
    ) {
        for (s, w) in [
            (&inout, "fr_mul_sub inout"),
            (&b, "fr_mul_sub b"),
            (&c, "fr_mul_sub c"),
        ] {
            s.check(n * FR_BYTES, w);
        }
        let p = FrEltParams {
            count: count_u32(n, "fr_mul_sub_from_mont"),
            _pad: [0; 3],
        };
        Self::dispatch(enc, &self.fr_mul_sub_from_mont, &[inout, b, c], &p, n);
    }

    /// `inout[i] += b[i]` for `i < n` (witness + fuzz).
    pub fn fr_add(&self, enc: &ComputeCommandEncoderRef, inout: BufSlice, b: BufSlice, n: u64) {
        inout.check(n * FR_BYTES, "fr_add inout");
        b.check(n * FR_BYTES, "fr_add b");
        let p = FrEltParams {
            count: count_u32(n, "fr_add"),
            _pad: [0; 3],
        };
        Self::dispatch(enc, &self.fr_add, &[inout, b], &p, n);
    }

    /// For each row `i < rows`: `out[c] = sum_j witness[coeffs[j].s] * coeffs[j].value` over
    /// `j in [indices[i], indices[i + 1])`, `c` the row's coefficients' `c`; empty rows write
    /// nothing, so `out` must be zero-filled first. `indices` holds `rows + 1` entries (relative to
    /// the `coeffs` slice). Bad coefficient data sets [witness_fault] bits in the u32 at `fault`
    /// (ORed; the caller zeroes it and reads it after the command buffer completes).
    #[allow(clippy::too_many_arguments)]
    pub fn witness_into_poly(
        &self,
        enc: &ComputeCommandEncoderRef,
        out: BufSlice,
        out_len: u64,
        coeffs: BufSlice,
        coeff_len: u64,
        indices: BufSlice,
        rows: u64,
        witness: BufSlice,
        witness_len: u64,
        fault: BufSlice,
    ) {
        out.check(out_len * FR_BYTES, "witness_into_poly out");
        coeffs.check(coeff_len * COEFF_BYTES, "witness_into_poly coeffs");
        indices.check((rows + 1) * 4, "witness_into_poly indices");
        witness.check(witness_len * FR_BYTES, "witness_into_poly witness");
        fault.check(4, "witness_into_poly fault");
        let p = WitnessPolyParams {
            rows: count_u32(rows, "witness_into_poly rows"),
            coeff_len: count_u32(coeff_len, "witness_into_poly coeffs"),
            witness_len: count_u32(witness_len, "witness_into_poly witness"),
            out_len: count_u32(out_len, "witness_into_poly out"),
        };
        Self::dispatch(
            enc,
            &self.witness_into_poly,
            &[out, coeffs, indices, witness, fault],
            &p,
            rows,
        );
    }

    /// The CUDA prover's fuzz: fills `out[0 .. 2 * blocks)` (every slot, unwritten ones with
    /// zero) from ChaCha8 blocks `0..blocks` of `state` ([chacha_state]; the prover uses the
    /// all-zero key and nonce). `blocks` = [fuzz_blocks]`(n)` for `n` scalars; the first `n`
    /// are used.
    pub fn chacha_fuzz(
        &self,
        enc: &ComputeCommandEncoderRef,
        out: BufSlice,
        state: &[u32; 16],
        blocks: u64,
    ) {
        out.check(2 * blocks * FR_BYTES, "chacha_fuzz out");
        let threads = blocks.div_ceil(FUZZ_WARP) * FUZZ_WARP;
        // idx + 32 is computed in 32 bits on the GPU.
        count_u32(2 * threads + FUZZ_WARP, "chacha_fuzz");
        let p = ChachaFuzzParams {
            state: *state,
            blocks: blocks as u32,
            threads: threads as u32,
            _pad: [0; 2],
        };
        Self::dispatch(enc, &self.chacha_fuzz, &[out], &p, threads);
    }

    /// `out[j] = scale * base^j` for `j < n` (twiddles: scale = 1; coset tables: scale = n^-1).
    pub fn fr_powers(
        &self,
        enc: &ComputeCommandEncoderRef,
        out: BufSlice,
        n: u64,
        base: &Fr,
        scale: &Fr,
    ) {
        out.check(n * FR_BYTES, "fr_powers out");
        let mut sq = [[0u32; 8]; 32];
        let mut t = *base;
        for e in sq.iter_mut() {
            *e = fr_to_words(&t);
            t.square_in_place();
        }
        let p = FrPowersParams {
            sq,
            scale: fr_to_words(scale),
            count: count_u32(n, "fr_powers"),
            _pad: [0; 3],
        };
        Self::dispatch(enc, &self.fr_powers, &[out], &p, n);
    }

    /// `out[i] = input[i] * scale` for `i < n` (`out` may be `input`).
    pub fn fr_scale(
        &self,
        enc: &ComputeCommandEncoderRef,
        out: BufSlice,
        input: BufSlice,
        n: u64,
        scale: &Fr,
    ) {
        out.check(n * FR_BYTES, "fr_scale out");
        input.check(n * FR_BYTES, "fr_scale in");
        let p = FrScaleParams {
            scale: fr_to_words(scale),
            count: count_u32(n, "fr_scale"),
            _pad: [0; 3],
        };
        Self::dispatch(enc, &self.fr_scale, &[out, input], &p, n);
    }
}

#[cfg(test)]
#[path = "small_tests.rs"]
mod tests;
