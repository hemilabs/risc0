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

//! GPU tests of the small Groth16 kernels (item G4) against arkworks (field ops) and rand_chacha
//! (the fuzz layout), bit-exact on every output word.

use std::ffi::c_void;

use ark_bn254::Fr;
use ark_ff::{AdditiveGroup, BigInt, BigInteger, FftField, Field, PrimeField, UniformRand};
use metal::{
    objc::{msg_send, sel, sel_impl},
    Buffer, CommandQueue, ComputeCommandEncoderRef, Device, MTLCommandBufferStatus,
    MTLResourceOptions,
};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::{ChaCha20Rng, ChaCha8Rng};

use super::*;

const N_RANDOM: usize = 100_000;

struct Gpu {
    device: Device,
    queue: CommandQueue,
    k: SmallKernels,
    /// The shared GPU test lock (`super::super::test_lock`), held for the test.
    _lock: std::sync::RwLockReadGuard<'static, ()>,
}

impl Gpu {
    fn new() -> Option<Self> {
        let _lock = super::super::test_lock::shared();
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping");
            return None;
        };
        assert!(!crate::METAL_LIB.is_empty(), "groth16 metallib is empty");
        for name in SMALL_KERNEL_NAMES {
            assert!(
                crate::METAL_KERNEL_NAMES.contains(name),
                "{name} not listed"
            );
        }
        let lib = device.new_library_with_data(crate::METAL_LIB).unwrap();
        let k = SmallKernels::new(&device, &lib).unwrap();
        let queue = device.new_command_queue();
        Some(Self {
            device,
            queue,
            k,
            _lock,
        })
    }

    fn buf(&self, data: &[u32]) -> Buffer {
        let one = [0u32];
        let data = if data.is_empty() { &one[..] } else { data };
        self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn zeros(&self, words: usize) -> Buffer {
        self.buf(&vec![0u32; words.max(1)])
    }

    fn read(buf: &Buffer, words: usize) -> Vec<u32> {
        unsafe { std::slice::from_raw_parts(buf.contents() as *const u32, words) }.to_vec()
    }

    fn write(buf: &Buffer, data: &[u32]) {
        assert!(std::mem::size_of_val(data) as u64 <= buf.length());
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut u32, data.len())
        }
    }

    /// Encode with `f` into one command buffer, commit, wait; returns GPU seconds.
    fn run(&self, f: impl FnOnce(&ComputeCommandEncoderRef)) -> f64 {
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        f(enc);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
        gpu_seconds(cb)
    }
}

// objc's `sel_impl!` checks `feature = "cargo-clippy"`, unknown to this crate's check-cfg.
#[allow(unexpected_cfgs)]
fn gpu_seconds(cb: &metal::CommandBufferRef) -> f64 {
    let (start, end): (f64, f64) =
        unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
    end - start
}

fn words(v: &[Fr]) -> Vec<u32> {
    v.iter().flat_map(fr_to_words).collect()
}

fn frs(w: &[u32]) -> Vec<Fr> {
    w.chunks(8).map(fr_from_words).collect()
}

/// The canonical integer of v as 8 u32 words (what `from_mont()` produces).
fn canonical_words(v: &Fr) -> [u32; 8] {
    let b = v.into_bigint();
    let mut w = [0u32; 8];
    for i in 0..4 {
        w[2 * i] = b.0[i] as u32;
        w[2 * i + 1] = (b.0[i] >> 32) as u32;
    }
    w
}

fn assert_words_eq(what: &str, got: &[u32], want: &[u32]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let bad: Vec<usize> = (0..got.len() / 8)
        .filter(|&i| got[8 * i..8 * i + 8] != want[8 * i..8 * i + 8])
        .collect();
    assert!(
        bad.is_empty(),
        "{what}: {} of {} elements differ, first at {}: got {:08x?} want {:08x?}",
        bad.len(),
        got.len() / 8,
        bad[0],
        &got[8 * bad[0]..8 * bad[0] + 8],
        &want[8 * bad[0]..8 * bad[0] + 8]
    );
}

/// Edge elements: as values (0, 1, 2, p-1, p-2, (p+-1)/2, 2^32-1, 2^64, 2^128, 2^253) and as raw
/// Montgomery limbs (the same integers read as Montgomery words, e.g. raw 1 = R^-1, raw p-1).
fn edges() -> Vec<Fr> {
    let p = Fr::MODULUS;
    let sub = |a: BigInt<4>, k: u64| {
        let mut r = a;
        r.sub_with_borrow(&BigInt::from(k));
        r
    };
    let mut half_lo = sub(p, 1);
    half_lo.div2();
    let mut half_hi = p;
    half_hi.add_with_carry(&BigInt::from(1u64));
    half_hi.div2();
    let pow2 = |k: u32| {
        let mut r = BigInt::<4>::zero();
        r.0[(k / 64) as usize] = 1u64 << (k % 64);
        r
    };
    let ints = [
        BigInt::zero(),
        BigInt::from(1u64),
        BigInt::from(2u64),
        sub(p, 1),
        sub(p, 2),
        half_lo,
        half_hi,
        BigInt::from(u32::MAX as u64),
        pow2(64),
        pow2(128),
        pow2(253),
    ];
    let mut v = Vec::new();
    for i in ints {
        v.push(Fr::from_bigint(i).unwrap());
        v.push(Fr::new_unchecked(i));
    }
    v
}

/// Edge pairs (all of edges x edges) followed by N_RANDOM random pairs.
fn operands(rng: &mut ChaCha20Rng) -> (Vec<Fr>, Vec<Fr>) {
    let e = edges();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for x in &e {
        for y in &e {
            a.push(*x);
            b.push(*y);
        }
    }
    for i in 0..N_RANDOM {
        let x = Fr::rand(rng);
        a.push(x);
        b.push(match i % 1000 {
            0 => x,
            1 => -x,
            _ => Fr::rand(rng),
        });
    }
    (a, b)
}

#[test]
fn word_conversion_matches_ark() {
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    for v in edges()
        .into_iter()
        .chain((0..1000).map(|_| Fr::rand(&mut rng)))
    {
        assert_eq!(fr_from_words(&fr_to_words(&v)), v);
    }
    // Montgomery one is R mod p.
    let one = fr_to_words(&Fr::ONE);
    assert_eq!(one[0], 0x4ffffffb);
}

#[test]
fn eltwise_vs_ark() {
    let Some(gpu) = Gpu::new() else { return };
    let mut rng = ChaCha20Rng::seed_from_u64(2);
    let (a, b) = operands(&mut rng);
    let c: Vec<Fr> = (0..a.len()).map(|_| Fr::rand(&mut rng)).collect();
    let n = a.len();
    // Offset slices (7 elements in) exercise the BufSlice offsets.
    const OFF: usize = 7;
    let pad = |v: &[Fr]| {
        let mut w = vec![0xdeadbeefu32; 8 * OFF];
        w.extend(words(v));
        w.extend([0xdeadbeefu32; 8]);
        w
    };
    let (ba, bb, bc) = (gpu.buf(&pad(&a)), gpu.buf(&pad(&b)), gpu.buf(&pad(&c)));
    let out = gpu.buf(&pad(&vec![Fr::ZERO; n]));
    fn at(buf: &Buffer) -> BufSlice<'_> {
        BufSlice::fr_at(buf, OFF as u64)
    }
    let read = |buf: &Buffer| {
        let w = Gpu::read(buf, 8 * (OFF + n + 1));
        assert!(
            w[..8 * OFF].iter().all(|&x| x == 0xdeadbeef),
            "prefix clobbered"
        );
        assert!(
            w[8 * (OFF + n)..].iter().all(|&x| x == 0xdeadbeef),
            "tail clobbered"
        );
        w[8 * OFF..8 * (OFF + n)].to_vec()
    };

    // fr_mul
    gpu.run(|e| gpu.k.fr_mul(e, at(&out), at(&ba), at(&bb), n as u64));
    let want: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
    assert_words_eq("fr_mul", &read(&out), &words(&want));

    // fr_mul in place (out = a)
    let ba2 = gpu.buf(&pad(&a));
    gpu.run(|e| gpu.k.fr_mul(e, at(&ba2), at(&ba2), at(&bb), n as u64));
    assert_words_eq("fr_mul in place", &read(&ba2), &words(&want));

    // fr_mul_sub_from_mont
    let inout = gpu.buf(&pad(&a));
    gpu.run(|e| {
        gpu.k
            .fr_mul_sub_from_mont(e, at(&inout), at(&bb), at(&bc), n as u64)
    });
    let want: Vec<u32> = (0..n)
        .flat_map(|i| canonical_words(&(a[i] * b[i] - c[i])))
        .collect();
    assert_words_eq("fr_mul_sub_from_mont", &read(&inout), &want);

    // fr_add
    let inout = gpu.buf(&pad(&a));
    gpu.run(|e| gpu.k.fr_add(e, at(&inout), at(&bb), n as u64));
    let want: Vec<Fr> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_words_eq("fr_add", &read(&inout), &words(&want));

    // fr_scale (out-of-place and in place), by an edge and a random scalar
    for s in [edges()[7], Fr::rand(&mut rng)] {
        gpu.run(|e| gpu.k.fr_scale(e, at(&out), at(&ba), n as u64, &s));
        let want: Vec<Fr> = a.iter().map(|x| *x * s).collect();
        assert_words_eq("fr_scale", &read(&out), &words(&want));
        let io = gpu.buf(&pad(&a));
        gpu.run(|e| gpu.k.fr_scale(e, at(&io), at(&io), n as u64, &s));
        assert_words_eq("fr_scale in place", &read(&io), &words(&want));
    }

    // Small and tail sizes (the threadgroup is 256 wide).
    for m in [1usize, 2, 255, 256, 257, 1000] {
        let o = gpu.buf(&pad(&vec![Fr::ZERO; n]));
        gpu.run(|e| gpu.k.fr_mul(e, at(&o), at(&ba), at(&bb), m as u64));
        let got = read(&o);
        let want: Vec<Fr> = a[..m].iter().zip(&b[..m]).map(|(x, y)| x * y).collect();
        assert_words_eq("fr_mul tail", &got[..8 * m], &words(&want));
        assert!(
            got[8 * m..].iter().all(|&x| x == 0),
            "fr_mul wrote past n = {m}"
        );
    }
    // n = 0 encodes nothing.
    gpu.run(|e| gpu.k.fr_mul(e, at(&out), at(&ba), at(&bb), 0));
}

#[test]
fn powers_vs_ark() {
    let Some(gpu) = Gpu::new() else { return };
    let mut rng = ChaCha20Rng::seed_from_u64(3);
    let max = (1usize << 16) + 3;
    let out = gpu.zeros(8 * (max + 1));
    let mut cases: Vec<(Fr, Fr, usize)> = vec![
        (Fr::ZERO, Fr::ONE, 5),
        (Fr::ONE, Fr::ONE, 300),
        (-Fr::ONE, Fr::ONE, 300),
        (Fr::rand(&mut rng), Fr::ONE, 1),
        (Fr::rand(&mut rng), Fr::ONE, 2),
        (Fr::rand(&mut rng), Fr::rand(&mut rng), 257),
        (Fr::rand(&mut rng), Fr::rand(&mut rng), max),
    ];
    // NTT tables: omega_n^j for j < n/2, and n^-1 * omega_2n^j for j < n.
    for lg in [1u64, 2, 10, 16] {
        let n = 1usize << lg;
        let w = Fr::get_root_of_unity(1 << lg).unwrap();
        let w2 = Fr::get_root_of_unity(2 << lg).unwrap();
        let ninv = Fr::from(n as u64).inverse().unwrap();
        cases.push((w, Fr::ONE, n / 2));
        cases.push((w.inverse().unwrap(), Fr::ONE, n / 2));
        cases.push((w2, ninv, n));
    }
    for (base, scale, n) in cases {
        Gpu::write(&out, &vec![0xa5a5a5a5u32; 8 * (max + 1)]);
        gpu.run(|e| {
            gpu.k
                .fr_powers(e, BufSlice::new(&out), n as u64, &base, &scale)
        });
        let mut want = Vec::with_capacity(n);
        let mut x = scale;
        for _ in 0..n {
            want.push(x);
            x *= base;
        }
        let got = Gpu::read(&out, 8 * (n + 1));
        assert_words_eq(&format!("fr_powers n={n}"), &got[..8 * n], &words(&want));
        assert!(
            got[8 * n..].iter().all(|&x| x == 0xa5a5a5a5),
            "fr_powers wrote past n"
        );
    }
    // Large exponents: the top table entries (j up to 2^22 - 1 uses sq[0..22]).
    let n = 1usize << 22;
    let big = gpu.zeros(8 * n);
    let base = Fr::rand(&mut rng);
    gpu.run(|e| {
        gpu.k
            .fr_powers(e, BufSlice::new(&big), n as u64, &base, &Fr::ONE)
    });
    let got = frs(&Gpu::read(&big, 8 * n));
    for j in [0usize, 1, 2, 3, 1 << 21, (1 << 21) + 1, n - 2, n - 1] {
        assert_eq!(got[j], base.pow([j as u64]), "fr_powers j={j}");
    }
    for j in 1..n {
        if j % 4099 == 0 {
            assert_eq!(got[j], got[j - 1] * base, "fr_powers chain at {j}");
        }
    }
}

/// The first `16 * blocks` words of rand_chacha's ChaCha8 stream with the all-zero key and nonce
/// (block i = words 16 i .. 16 i + 16).
fn chacha8_stream(blocks: usize) -> Vec<u32> {
    let mut rng = ChaCha8Rng::from_seed([0u8; 32]);
    (0..16 * blocks).map(|_| rng.next_u32()).collect()
}

/// The CUDA kernel's output for `blocks` blocks of `stream`, written from its source formula:
/// idx = lane + wid * 64, v0 at idx, v1 at idx + 32 if < 2 * blocks, l[7] &= 0x1FFFFFFF, all
/// other slots zero (the bzero before the launch).
fn fuzz_reference(stream: &[u32], blocks: usize) -> Vec<u32> {
    let total = 2 * blocks;
    let mut out = vec![0u32; 8 * total];
    for i in 0..blocks {
        let (lane, wid) = (i % 32, i / 32);
        let idx = lane + wid * 64;
        let mut v0: [u32; 8] = stream[16 * i..16 * i + 8].try_into().unwrap();
        let mut v1: [u32; 8] = stream[16 * i + 8..16 * i + 16].try_into().unwrap();
        v0[7] &= 0x1FFFFFFF;
        v1[7] &= 0x1FFFFFFF;
        out[8 * idx..8 * idx + 8].copy_from_slice(&v0);
        if idx + 32 < total {
            out[8 * (idx + 32)..8 * (idx + 32) + 8].copy_from_slice(&v1);
        }
    }
    out
}

#[test]
fn chacha_state_matches_cuda_init() {
    let s = chacha_state(&[0; 32], &[0; 12]);
    assert_eq!(&s[..4], &[0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]);
    assert!(s[4..].iter().all(|&w| w == 0));
    let mut key = [0u8; 32];
    key[0] = 1;
    key[31] = 0x80;
    let s = chacha_state(&key, &[2; 12]);
    assert_eq!((s[4], s[11], s[12], s[13]), (1, 0x8000_0000, 0, 0x02020202));
}

#[test]
fn fuzz_vs_chacha8_reference() {
    let Some(gpu) = Gpu::new() else { return };
    // Every scalar count 1..=5000 (blocks 1..=2500: whole and partial warps), garbage-prefilled
    // so the zero tails are checked, plus a guard past the end.
    let max_blocks = fuzz_blocks(5000) as usize;
    let stream = chacha8_stream(max_blocks);
    let out = gpu.zeros(8 * (2 * max_blocks + 64));
    let state = chacha_state(&[0; 32], &[0; 12]);
    for n in 1..=5000u64 {
        let blocks = fuzz_blocks(n) as usize;
        assert_eq!(blocks, (n as usize).div_ceil(2));
        Gpu::write(&out, &vec![0x5a5a5a5au32; 8 * (2 * max_blocks + 64)]);
        gpu.run(|e| {
            gpu.k
                .chacha_fuzz(e, BufSlice::new(&out), &state, blocks as u64)
        });
        let got = Gpu::read(&out, 8 * (2 * blocks + 64));
        let want = fuzz_reference(&stream, blocks);
        assert_words_eq(&format!("chacha_fuzz n={n}"), &got[..16 * blocks], &want);
        assert!(
            got[16 * blocks..].iter().all(|&w| w == 0x5a5a5a5a),
            "chacha_fuzz n={n} wrote past 2 * blocks"
        );
    }
    // Large: 2^20 + 5 scalars; also every value canonical (< r).
    let n = (1u64 << 20) + 5;
    let blocks = fuzz_blocks(n) as usize;
    let big = gpu.zeros(16 * blocks);
    gpu.run(|e| {
        gpu.k
            .chacha_fuzz(e, BufSlice::new(&big), &state, blocks as u64)
    });
    let got = Gpu::read(&big, 16 * blocks);
    assert_words_eq(
        "chacha_fuzz large",
        &got,
        &fuzz_reference(&chacha8_stream(blocks), blocks),
    );
    for v in got.chunks(8) {
        let f = fr_from_words(v);
        assert!(f.0 < Fr::MODULUS);
    }
}

struct WitnessCase {
    coeffs: Vec<Coeff>,
    indices: Vec<u32>,
    witness: Vec<u32>,
    out_len: usize,
}

/// Random rows with 0..=6 coefficients (0 = an empty row), one long row, outputs permuted.
fn witness_case(rng: &mut ChaCha20Rng, rows: usize, witness_len: usize) -> WitnessCase {
    let out_len = rows + 9;
    let mut targets: Vec<u32> = (0..out_len as u32).collect();
    for i in (1..targets.len()).rev() {
        targets.swap(i, rng.gen_range(0..=i));
    }
    let witness: Vec<u32> = (0..witness_len)
        .flat_map(|_| fr_to_words(&Fr::rand(rng)))
        .collect();
    let mut coeffs = Vec::new();
    let mut indices = vec![0u32];
    for (r, &target) in targets.iter().take(rows).enumerate() {
        let count = if r == rows / 2 {
            1000
        } else {
            rng.gen_range(0..=6)
        };
        for _ in 0..count {
            let value = match rng.gen_range(0..20) {
                0 => fr_to_words(&Fr::ZERO),
                1 => fr_to_words(&-Fr::ONE),
                _ => fr_to_words(&Fr::rand(rng)),
            };
            coeffs.push(Coeff {
                m: 0,
                c: target,
                s: rng.gen_range(0..witness_len as u32),
                pad: 0,
                value,
            });
        }
        indices.push(coeffs.len() as u32);
    }
    WitnessCase {
        coeffs,
        indices,
        witness,
        out_len,
    }
}

fn witness_expected(case: &WitnessCase) -> Vec<u32> {
    let mut out = vec![Fr::ZERO; case.out_len];
    for r in 0..case.indices.len() - 1 {
        let (s, e) = (case.indices[r] as usize, case.indices[r + 1] as usize);
        // Empty rows and rows the kernel rejects (bad range or index) write nothing.
        let wl = case.witness.len() / 8;
        if s >= e
            || e > case.coeffs.len()
            || case.coeffs[s..e]
                .iter()
                .any(|c| c.s as usize >= wl || c.c as usize >= case.out_len)
        {
            continue;
        }
        let mut sum = Fr::ZERO;
        for c in &case.coeffs[s..e] {
            let w = fr_from_words(&case.witness[8 * c.s as usize..8 * c.s as usize + 8]);
            sum += w * fr_from_words(&c.value);
        }
        out[case.coeffs[e - 1].c as usize] = sum;
    }
    words(&out)
}

fn coeff_words(c: &[Coeff]) -> Vec<u32> {
    c.iter()
        .flat_map(|c| {
            let mut w = vec![c.m, c.c, c.s, c.pad];
            w.extend(c.value);
            w
        })
        .collect()
}

/// Runs witness_into_poly over `case` (coefficients and indices placed at offsets inside larger
/// buffers, as the prover's B part is); returns (out words, fault).
fn run_witness(gpu: &Gpu, case: &WitnessCase) -> (Vec<u32>, u32) {
    const COFF: usize = 3; // coefficients before the slice
    const IOFF: usize = 5; // indices before the slice
    let mut cw = vec![0xffffffffu32; 12 * COFF];
    cw.extend(coeff_words(&case.coeffs));
    let mut iw = vec![0xffffffffu32; IOFF];
    iw.extend(&case.indices);
    let (bc, bi, bw) = (gpu.buf(&cw), gpu.buf(&iw), gpu.buf(&case.witness));
    let out = gpu.zeros(8 * case.out_len);
    let fault = gpu.zeros(1);
    gpu.run(|e| {
        gpu.k.witness_into_poly(
            e,
            BufSlice::new(&out),
            case.out_len as u64,
            BufSlice::at(&bc, COFF as u64 * COEFF_BYTES),
            case.coeffs.len() as u64,
            BufSlice::at(&bi, IOFF as u64 * 4),
            case.indices.len() as u64 - 1,
            BufSlice::new(&bw),
            case.witness.len() as u64 / 8,
            BufSlice::new(&fault),
        )
    });
    (Gpu::read(&out, 8 * case.out_len), Gpu::read(&fault, 1)[0])
}

#[test]
fn witness_into_poly_vs_ark() {
    let Some(gpu) = Gpu::new() else { return };
    let mut rng = ChaCha20Rng::seed_from_u64(4);
    for (rows, wl) in [(1usize, 1usize), (2, 3), (257, 50), (20_000, 3000)] {
        let case = witness_case(&mut rng, rows, wl);
        let (got, fault) = run_witness(&gpu, &case);
        assert_eq!(fault, 0, "rows={rows}");
        assert_words_eq(
            &format!("witness_into_poly rows={rows}"),
            &got,
            &witness_expected(&case),
        );
    }
}

#[test]
fn witness_into_poly_faults() {
    let Some(gpu) = Gpu::new() else { return };
    let mut rng = ChaCha20Rng::seed_from_u64(5);
    let _ = &mut rng;
    let base = witness_case(&mut ChaCha20Rng::seed_from_u64(5), 64, 40);
    let first_nonempty = (0..64)
        .find(|&r| base.indices[r + 1] > base.indices[r] && r != 32)
        .unwrap();
    let check_others = |case: &WitnessCase, got: &[u32], skip_row: Option<usize>| {
        let want = witness_expected(case);
        for r in 0..64 {
            let (s, e) = (case.indices[r] as usize, case.indices[r + 1] as usize);
            if Some(r) == skip_row || s >= e || e > case.coeffs.len() {
                continue;
            }
            let c = case.coeffs[e - 1].c as usize;
            assert_eq!(got[8 * c..8 * c + 8], want[8 * c..8 * c + 8], "row {r}");
        }
    };

    // s out of range
    let mut case = witness_case(&mut ChaCha20Rng::seed_from_u64(5), 64, 40);
    let j = case.indices[first_nonempty] as usize;
    case.coeffs[j].s = 40;
    let (got, fault) = run_witness(&gpu, &case);
    assert_eq!(fault, witness_fault::BAD_INDEX);
    check_others(&case, &got, Some(first_nonempty));
    // The rejected row wrote nothing: the whole output is the reference's.
    assert_words_eq("bad s", &got, &witness_expected(&case));

    // c out of range
    let mut case = witness_case(&mut ChaCha20Rng::seed_from_u64(5), 64, 40);
    case.coeffs[j].c = case.out_len as u32;
    let (got, fault) = run_witness(&gpu, &case);
    assert_eq!(fault, witness_fault::BAD_INDEX);
    assert_words_eq("bad c", &got, &witness_expected(&case));

    // indices decreasing / past the end
    let mut case = witness_case(&mut ChaCha20Rng::seed_from_u64(5), 64, 40);
    let last = case.indices.len() - 1;
    case.indices[last] = case.coeffs.len() as u32 + 1;
    let (got, fault) = run_witness(&gpu, &case);
    assert_eq!(fault, witness_fault::BAD_RANGE);
    check_others(&case, &got, Some(63));
    assert_words_eq("bad range", &got, &witness_expected(&case));

    // non-canonical value (p itself): flagged, other rows exact
    let mut case = witness_case(&mut ChaCha20Rng::seed_from_u64(5), 64, 40);
    let p = Fr::MODULUS;
    for i in 0..4 {
        case.coeffs[j].value[2 * i] = p.0[i] as u32;
        case.coeffs[j].value[2 * i + 1] = (p.0[i] >> 32) as u32;
    }
    let (got, fault) = run_witness(&gpu, &case);
    assert_eq!(fault, witness_fault::NON_CANONICAL);
    check_others(&case, &got, Some(first_nonempty));
}

/// GPU time per kernel at 2^lg elements (lg from RISC0_G16_SMALL_LG, default 20) and the
/// device's current_allocated_size; prints, asserts correctness of spot values only.
#[test]
fn small_throughput() {
    let Some(gpu) = Gpu::new() else { return };
    let lg: u32 = std::env::var("RISC0_G16_SMALL_LG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let n = 1usize << lg;
    let mut rng = ChaCha20Rng::seed_from_u64(6);
    let a: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let b: Vec<Fr> = (0..n).map(|_| Fr::rand(&mut rng)).collect();
    let alloc0 = gpu.device.current_allocated_size();
    let (ba, bb, bc) = (
        gpu.buf(&words(&a)),
        gpu.buf(&words(&b)),
        gpu.buf(&words(&b)),
    );
    let out = gpu.zeros(8 * n);
    let best = |f: &dyn Fn(&ComputeCommandEncoderRef)| -> f64 {
        (0..3).map(|_| gpu.run(f)).fold(f64::MAX, f64::min) * 1e3
    };
    let s = |x| BufSlice::new(x);
    let base = Fr::rand(&mut rng);
    let t_mul = best(&|e| gpu.k.fr_mul(e, s(&out), s(&ba), s(&bb), n as u64));
    let t_add = best(&|e| gpu.k.fr_add(e, s(&out), s(&bb), n as u64));
    let t_msf = best(&|e| {
        gpu.k
            .fr_mul_sub_from_mont(e, s(&out), s(&bb), s(&bc), n as u64)
    });
    let t_scale = best(&|e| gpu.k.fr_scale(e, s(&out), s(&ba), n as u64, &base));
    let t_pow = best(&|e| gpu.k.fr_powers(e, s(&out), n as u64, &base, &Fr::ONE));
    assert_eq!(
        fr_from_words(&Gpu::read(&out, 8 * n)[8 * (n - 1)..]),
        base.pow([n as u64 - 1])
    );
    let blocks = fuzz_blocks(n as u64);
    let t_fuzz = best(&|e| {
        gpu.k
            .chacha_fuzz(e, s(&out), &chacha_state(&[0; 32], &[0; 12]), blocks)
    });
    // witness_into_poly: n/2 rows of 4 coefficients over an n/4 witness (A-matrix-like density).
    let rows = n / 2;
    let wl = n / 4;
    let mut cw = Vec::with_capacity(12 * rows * 4);
    let mut iw = vec![0u32];
    for r in 0..rows {
        for _ in 0..4 {
            cw.extend([0, r as u32, rng.gen_range(0..wl as u32), 0]);
            cw.extend(fr_to_words(&a[rng.gen_range(0..n)]));
        }
        iw.push(4 * (r as u32 + 1));
    }
    let (bcf, bi) = (gpu.buf(&cw), gpu.buf(&iw));
    let fault = gpu.zeros(1);
    let t_wip = best(&|e| {
        gpu.k.witness_into_poly(
            e,
            s(&out),
            rows as u64,
            s(&bcf),
            4 * rows as u64,
            s(&bi),
            rows as u64,
            s(&ba),
            wl as u64,
            s(&fault),
        )
    });
    assert_eq!(Gpu::read(&fault, 1)[0], 0);
    let peak = gpu.device.current_allocated_size();
    eprintln!(
        "small kernels at n = 2^{lg}: fr_mul {t_mul:.2} ms, fr_add {t_add:.2} ms, \
         fr_mul_sub_from_mont {t_msf:.2} ms, fr_scale {t_scale:.2} ms, fr_powers {t_pow:.2} ms, \
         chacha_fuzz ({blocks} blocks) {t_fuzz:.2} ms, witness_into_poly ({rows} rows x 4) \
         {t_wip:.2} ms; current_allocated_size {:.1} MiB (before buffers {:.1} MiB)",
        peak as f64 / (1 << 20) as f64,
        alloc0 as f64 / (1 << 20) as f64
    );
}
