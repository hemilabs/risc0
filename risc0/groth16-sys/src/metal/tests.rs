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

//! GPU equivalence tests for `kernels/metal/bn254.h` (item G2), run through the test kernels in
//! `kernels/metal/bn254_test.metal` with metal-rs.
//!
//! The reference is self-contained (no ark / num-bigint dependency is reachable from this crate):
//! 4 x 64-bit-limb integers, Montgomery multiplication with u128 products (a different limb size
//! and code path than the GPU's 8 x 32-bit CIOS), cross-checked against plain double-and-add
//! modular multiplication; Montgomery constants are derived from the moduli at run time, not
//! copied from the header. Curve results are compared in affine coordinates computed with
//! chord-and-tangent formulas (batch inversion), independent of the XYZZ formulas under test.
//!
//! Every op runs on [N_RANDOM] random cases plus hand-picked edge cases.

#![allow(clippy::needless_range_loop)]

use std::{ffi::c_void, sync::OnceLock};

use metal::{
    objc::{msg_send, sel, sel_impl},
    Buffer, CommandQueue, Device, Library, MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};

/// Random cases per op (the acceptance floor is 10^5).
const N_RANDOM: usize = 100_000;

// ------------------------------------------------------------------------------------------------
// Deterministic RNG (splitmix64)
// ------------------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn big(&mut self) -> L {
        [self.next(), self.next(), self.next(), self.next()]
    }
}

// ------------------------------------------------------------------------------------------------
// 256-bit integers, little-endian u64 limbs
// ------------------------------------------------------------------------------------------------

type L = [u64; 4];

fn big_add(a: &L, b: &L) -> (L, u64) {
    let mut r = [0u64; 4];
    let mut c = 0u128;
    for i in 0..4 {
        let s = a[i] as u128 + b[i] as u128 + c;
        r[i] = s as u64;
        c = s >> 64;
    }
    (r, c as u64)
}

fn big_sub(a: &L, b: &L) -> (L, u64) {
    let mut r = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
        let (d1, b1) = a[i].overflowing_sub(b[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        r[i] = d2;
        borrow = (b1 | b2) as u64;
    }
    (r, borrow)
}

fn big_geq(a: &L, b: &L) -> bool {
    for i in (0..4).rev() {
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

fn big_bit(a: &L, i: usize) -> bool {
    (a[i / 64] >> (i % 64)) & 1 == 1
}

fn big_from_u64(v: u64) -> L {
    [v, 0, 0, 0]
}

/// Decimal string to a 256-bit integer (panics on overflow).
fn big_from_dec(s: &str) -> L {
    let mut r = [0u64; 4];
    for ch in s.bytes() {
        let d = (ch - b'0') as u128;
        let mut c = d;
        for limb in r.iter_mut() {
            let v = (*limb as u128) * 10 + c;
            *limb = v as u64;
            c = v >> 64;
        }
        assert_eq!(c, 0, "decimal overflow");
    }
    r
}

/// Shift left by `n` < 64 bits (drops overflow).
fn big_shl(a: &L, n: u32) -> L {
    if n == 0 {
        return *a;
    }
    let mut r = [0u64; 4];
    for i in 0..4 {
        r[i] = a[i] << n;
        if i > 0 {
            r[i] |= a[i - 1] >> (64 - n);
        }
    }
    r
}

fn big_shr1(a: &L) -> L {
    let mut r = [0u64; 4];
    for i in 0..4 {
        r[i] = a[i] >> 1;
        if i < 3 {
            r[i] |= a[i + 1] << 63;
        }
    }
    r
}

fn pow2(k: usize) -> L {
    let mut r = [0u64; 4];
    r[k / 64] = 1 << (k % 64);
    r
}

fn words_of(a: &L, out: &mut Vec<u32>) {
    for v in a {
        out.push(*v as u32);
        out.push((*v >> 32) as u32);
    }
}

fn from_words(w: &[u32]) -> L {
    let mut r = [0u64; 4];
    for i in 0..4 {
        r[i] = w[2 * i] as u64 | ((w[2 * i + 1] as u64) << 32);
    }
    r
}

// ------------------------------------------------------------------------------------------------
// Reference prime field (Montgomery form, R = 2^256)
// ------------------------------------------------------------------------------------------------

struct FieldRef {
    p: L,
    /// -p^-1 mod 2^64
    m0: u64,
    /// R mod p (Montgomery one)
    r: L,
    /// R^2 mod p
    r2: L,
}

impl FieldRef {
    fn new(p: L) -> Self {
        // Every constant derived here from p alone.
        let mut f = Self {
            p,
            m0: 0,
            r: [0; 4],
            r2: [0; 4],
        };
        let mut x = big_from_u64(1);
        for _ in 0..256 {
            x = f.add(&x, &x);
        }
        f.r = x;
        for _ in 0..256 {
            x = f.add(&x, &x);
        }
        f.r2 = x;
        let mut inv = 1u64;
        for _ in 0..7 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(p[0].wrapping_mul(inv)));
        }
        assert_eq!(p[0].wrapping_mul(inv), 1);
        f.m0 = inv.wrapping_neg();
        f
    }

    fn reduce(&self, v: &L) -> L {
        let mut v = *v;
        while big_geq(&v, &self.p) {
            v = big_sub(&v, &self.p).0;
        }
        v
    }

    fn add(&self, a: &L, b: &L) -> L {
        let (s, c) = big_add(a, b);
        if c != 0 || big_geq(&s, &self.p) {
            big_sub(&s, &self.p).0
        } else {
            s
        }
    }

    fn sub(&self, a: &L, b: &L) -> L {
        let (d, br) = big_sub(a, b);
        if br != 0 {
            big_add(&d, &self.p).0
        } else {
            d
        }
    }

    fn neg(&self, a: &L) -> L {
        self.sub(&[0; 4], a)
    }

    /// a * b * R^-1 mod p (CIOS over u64 limbs). Canonical for a * b < p R.
    fn mul(&self, a: &L, b: &L) -> L {
        let p = &self.p;
        let mut t = [0u64; 6];
        for i in 0..4 {
            let mut c = 0u128;
            for j in 0..4 {
                let s = t[j] as u128 + (a[j] as u128) * (b[i] as u128) + c;
                t[j] = s as u64;
                c = s >> 64;
            }
            let s = t[4] as u128 + c;
            t[4] = s as u64;
            t[5] = (s >> 64) as u64;
            let m = t[0].wrapping_mul(self.m0);
            let s = t[0] as u128 + (m as u128) * (p[0] as u128);
            let mut c = s >> 64;
            for j in 1..4 {
                let s = t[j] as u128 + (m as u128) * (p[j] as u128) + c;
                t[j - 1] = s as u64;
                c = s >> 64;
            }
            let s = t[4] as u128 + c;
            t[3] = s as u64;
            t[4] = t[5] + (s >> 64) as u64;
        }
        let r = [t[0], t[1], t[2], t[3]];
        if t[4] != 0 || big_geq(&r, p) {
            big_sub(&r, p).0
        } else {
            r
        }
    }

    fn pow(&self, a: &L, e: &L) -> L {
        let mut r = self.r;
        for i in (0..256).rev() {
            r = self.mul(&r, &r);
            if big_bit(e, i) {
                r = self.mul(&r, a);
            }
        }
        r
    }

    fn inv(&self, a: &L) -> L {
        let e = big_sub(&self.p, &big_from_u64(2)).0;
        self.pow(a, &e)
    }

    /// Any 256-bit v -> (v mod p) R mod p.
    fn to_mont(&self, v: &L) -> L {
        self.mul(&self.reduce(v), &self.r2)
    }

    /// Any 256-bit v -> v R^-1 mod p.
    fn mont_reduce(&self, v: &L) -> L {
        self.mul(&self.reduce(v), &big_from_u64(1))
    }

    /// a * b mod p by double-and-add (independent of Montgomery arithmetic).
    fn slow_mulmod(&self, a: &L, b: &L) -> L {
        let mut acc = [0u64; 4];
        for i in (0..256).rev() {
            acc = self.add(&acc, &acc);
            if big_bit(b, i) {
                acc = self.add(&acc, a);
            }
        }
        acc
    }

    /// Uniform canonical value.
    fn random(&self, rng: &mut Rng) -> L {
        let top_mask = u64::MAX >> self.p[3].leading_zeros();
        loop {
            let mut v = rng.big();
            v[3] &= top_mask;
            if !big_geq(&v, &self.p) {
                return v;
            }
        }
    }
}

const FP_MODULUS_DEC: &str =
    "21888242871839275222246405745257275088696311157297823662689037894645226208583";
const FR_MODULUS_DEC: &str =
    "21888242871839275222246405745257275088548364400416034343698204186575808495617";

fn fp() -> &'static FieldRef {
    static F: OnceLock<FieldRef> = OnceLock::new();
    F.get_or_init(|| FieldRef::new(big_from_dec(FP_MODULUS_DEC)))
}

fn fr() -> &'static FieldRef {
    static F: OnceLock<FieldRef> = OnceLock::new();
    F.get_or_init(|| FieldRef::new(big_from_dec(FR_MODULUS_DEC)))
}

// ------------------------------------------------------------------------------------------------
// Reference field trait for the curve code (Fp and Fp2, elements in Montgomery form)
// ------------------------------------------------------------------------------------------------

trait RF: Copy + PartialEq + std::fmt::Debug {
    const WORDS: usize;
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn neg(&self) -> Self;
    fn inv(&self) -> Self;
    fn is_zero(&self) -> bool;
    fn random(rng: &mut Rng) -> Self;
    fn write(&self, out: &mut Vec<u32>);
    fn read(w: &[u32]) -> Self;
    fn sqr(&self) -> Self {
        self.mul(self)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct RFp(L);

impl RF for RFp {
    const WORDS: usize = 8;
    fn zero() -> Self {
        Self([0; 4])
    }
    fn one() -> Self {
        Self(fp().r)
    }
    fn add(&self, o: &Self) -> Self {
        Self(fp().add(&self.0, &o.0))
    }
    fn sub(&self, o: &Self) -> Self {
        Self(fp().sub(&self.0, &o.0))
    }
    fn mul(&self, o: &Self) -> Self {
        Self(fp().mul(&self.0, &o.0))
    }
    fn neg(&self) -> Self {
        Self(fp().neg(&self.0))
    }
    fn inv(&self) -> Self {
        Self(fp().inv(&self.0))
    }
    fn is_zero(&self) -> bool {
        self.0 == [0; 4]
    }
    fn random(rng: &mut Rng) -> Self {
        Self(fp().random(rng))
    }
    fn write(&self, out: &mut Vec<u32>) {
        words_of(&self.0, out)
    }
    fn read(w: &[u32]) -> Self {
        Self(from_words(w))
    }
}

/// c0 + c1 u, u^2 = -1.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct RFp2(RFp, RFp);

impl RF for RFp2 {
    const WORDS: usize = 16;
    fn zero() -> Self {
        Self(RFp::zero(), RFp::zero())
    }
    fn one() -> Self {
        Self(RFp::one(), RFp::zero())
    }
    fn add(&self, o: &Self) -> Self {
        Self(self.0.add(&o.0), self.1.add(&o.1))
    }
    fn sub(&self, o: &Self) -> Self {
        Self(self.0.sub(&o.0), self.1.sub(&o.1))
    }
    // Schoolbook (the GPU uses Karatsuba).
    fn mul(&self, o: &Self) -> Self {
        Self(
            self.0.mul(&o.0).sub(&self.1.mul(&o.1)),
            self.0.mul(&o.1).add(&self.1.mul(&o.0)),
        )
    }
    fn neg(&self) -> Self {
        Self(self.0.neg(), self.1.neg())
    }
    fn inv(&self) -> Self {
        let n = self.0.sqr().add(&self.1.sqr()).inv();
        Self(self.0.mul(&n), self.1.mul(&n).neg())
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero() && self.1.is_zero()
    }
    fn random(rng: &mut Rng) -> Self {
        Self(RFp::random(rng), RFp::random(rng))
    }
    fn write(&self, out: &mut Vec<u32>) {
        self.0.write(out);
        self.1.write(out);
    }
    fn read(w: &[u32]) -> Self {
        Self(RFp::read(&w[..8]), RFp::read(&w[8..16]))
    }
}

/// Batch inversion (Montgomery's trick); zeros map to zero.
fn batch_inv<F: RF>(v: &[F]) -> Vec<F> {
    let mut prefix = Vec::with_capacity(v.len());
    let mut acc = F::one();
    for x in v {
        prefix.push(acc);
        if !x.is_zero() {
            acc = acc.mul(x);
        }
    }
    let mut inv = acc.inv();
    let mut out = vec![F::zero(); v.len()];
    for i in (0..v.len()).rev() {
        if v[i].is_zero() {
            continue;
        }
        out[i] = inv.mul(&prefix[i]);
        inv = inv.mul(&v[i]);
    }
    out
}

// ------------------------------------------------------------------------------------------------
// Reference curve arithmetic (affine, chord and tangent, a = 0)
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Aff<F> {
    x: F,
    y: F,
    inf: bool,
}

impl<F: RF> Aff<F> {
    fn inf() -> Self {
        Self {
            x: F::zero(),
            y: F::zero(),
            inf: true,
        }
    }
    fn neg(&self) -> Self {
        if self.inf {
            *self
        } else {
            Self {
                x: self.x,
                y: self.y.neg(),
                inf: false,
            }
        }
    }
    fn on_curve(&self, b: &F) -> bool {
        self.inf || self.y.sqr() == self.x.sqr().mul(&self.x).add(b)
    }
    /// GPU affine words; infinity is (0, 0).
    fn write(&self, out: &mut Vec<u32>) {
        self.x.write(out);
        self.y.write(out);
    }
}

/// Elementwise p[i] + q[i].
fn batch_add<F: RF>(p: &[Aff<F>], q: &[Aff<F>]) -> Vec<Aff<F>> {
    #[derive(Clone, Copy)]
    enum Case {
        P,
        Q,
        Inf,
        Add,
        Dbl,
    }
    let n = p.len();
    let mut cases = Vec::with_capacity(n);
    let mut dens = Vec::with_capacity(n);
    for i in 0..n {
        let (a, b) = (&p[i], &q[i]);
        let (c, d) = if a.inf {
            (Case::Q, F::zero())
        } else if b.inf {
            (Case::P, F::zero())
        } else if a.x == b.x {
            if a.y == b.y && !a.y.is_zero() {
                (Case::Dbl, a.y.add(&a.y))
            } else {
                (Case::Inf, F::zero())
            }
        } else {
            (Case::Add, b.x.sub(&a.x))
        };
        cases.push(c);
        dens.push(d);
    }
    let inv = batch_inv(&dens);
    (0..n)
        .map(|i| {
            let (a, b) = (&p[i], &q[i]);
            let lambda = match cases[i] {
                Case::P => return *a,
                Case::Q => return *b,
                Case::Inf => return Aff::inf(),
                Case::Add => b.y.sub(&a.y).mul(&inv[i]),
                Case::Dbl => {
                    let x2 = a.x.sqr();
                    x2.add(&x2).add(&x2).mul(&inv[i])
                }
            };
            let x3 = lambda.sqr().sub(&a.x).sub(&b.x);
            let y3 = lambda.mul(&a.x.sub(&x3)).sub(&a.y);
            Aff {
                x: x3,
                y: y3,
                inf: false,
            }
        })
        .collect()
}

/// Elementwise k[i] * p[i] for the low `bits` bits of k[i] (double and add).
fn batch_mul<F: RF>(p: &[Aff<F>], k: &[u64], bits: usize) -> Vec<Aff<F>> {
    let mut acc = vec![Aff::inf(); p.len()];
    for bit in (0..bits).rev() {
        acc = batch_add(&acc, &acc);
        let sel: Vec<_> = (0..p.len())
            .map(|i| {
                if (k[i] >> bit) & 1 == 1 {
                    p[i]
                } else {
                    Aff::inf()
                }
            })
            .collect();
        acc = batch_add(&acc, &sel);
    }
    acc
}

/// Xyzz words (X, Y, ZZ, ZZZ) of `a` scaled by `lambda` (infinity: all zero).
fn write_xyzz<F: RF>(a: &Aff<F>, lambda: &F, out: &mut Vec<u32>) {
    if a.inf {
        out.extend(std::iter::repeat_n(0, 4 * F::WORDS));
        return;
    }
    let l2 = lambda.sqr();
    let l3 = l2.mul(lambda);
    a.x.mul(&l2).write(out);
    a.y.mul(&l3).write(out);
    l2.write(out);
    l3.write(out);
}

/// Jacobian words (X, Y, Z) of `a` with Z = `z` (infinity: all zero).
fn write_jacobian<F: RF>(a: &Aff<F>, z: &F, out: &mut Vec<u32>) {
    if a.inf {
        out.extend(std::iter::repeat_n(0, 3 * F::WORDS));
        return;
    }
    let z2 = z.sqr();
    a.x.mul(&z2).write(out);
    a.y.mul(&z2.mul(z)).write(out);
    z.write(out);
}

/// Decode GPU Xyzz words to affine. Errors on a broken ZZ^3 = ZZZ^2 invariant.
fn read_xyzz<F: RF>(w: &[u32]) -> Vec<Result<Aff<F>, String>> {
    let n = w.len() / (4 * F::WORDS);
    let get = |i: usize, c: usize| F::read(&w[(4 * i + c) * F::WORDS..]);
    let zz: Vec<F> = (0..n).map(|i| get(i, 2)).collect();
    let zzz: Vec<F> = (0..n).map(|i| get(i, 3)).collect();
    let izz = batch_inv(&zz);
    let izzz = batch_inv(&zzz);
    (0..n)
        .map(|i| {
            if zz[i].is_zero() {
                return Ok(Aff::inf());
            }
            if zz[i].sqr().mul(&zz[i]) != zzz[i].sqr() {
                return Err("ZZ^3 != ZZZ^2".into());
            }
            Ok(Aff {
                x: get(i, 0).mul(&izz[i]),
                y: get(i, 1).mul(&izzz[i]),
                inf: false,
            })
        })
        .collect()
}

fn read_jacobian<F: RF>(w: &[u32]) -> Vec<Aff<F>> {
    let n = w.len() / (3 * F::WORDS);
    let get = |i: usize, c: usize| F::read(&w[(3 * i + c) * F::WORDS..]);
    let z: Vec<F> = (0..n).map(|i| get(i, 2)).collect();
    let iz = batch_inv(&z);
    (0..n)
        .map(|i| {
            if z[i].is_zero() {
                return Aff::inf();
            }
            let iz2 = iz[i].sqr();
            Aff {
                x: get(i, 0).mul(&iz2),
                y: get(i, 1).mul(&iz2.mul(&iz[i])),
                inf: false,
            }
        })
        .collect()
}

fn read_affine<F: RF>(w: &[u32]) -> Vec<Aff<F>> {
    w.chunks(2 * F::WORDS)
        .map(|c| {
            let (x, y) = (F::read(&c[..F::WORDS]), F::read(&c[F::WORDS..]));
            Aff {
                x,
                y,
                inf: x.is_zero() && y.is_zero(),
            }
        })
        .collect()
}

trait Curve: RF {
    const NAME: &'static str;
    const KERNEL: &'static str;
    fn b() -> Self;
    fn generator() -> Aff<Self>;
}

fn mont_fp(dec: &str) -> RFp {
    RFp(fp().to_mont(&big_from_dec(dec)))
}

impl Curve for RFp {
    const NAME: &'static str = "G1";
    const KERNEL: &'static str = "bn254_test_g1_op";
    fn b() -> Self {
        mont_fp("3")
    }
    fn generator() -> Aff<Self> {
        Aff {
            x: mont_fp("1"),
            y: mont_fp("2"),
            inf: false,
        }
    }
}

impl Curve for RFp2 {
    const NAME: &'static str = "G2";
    const KERNEL: &'static str = "bn254_test_g2_op";
    // The twist: b' = 3 / (9 + u).
    fn b() -> Self {
        RFp2(mont_fp("3"), RFp::zero()).mul(&RFp2(mont_fp("9"), mont_fp("1")).inv())
    }
    // EIP-197 G2 generator.
    fn generator() -> Aff<Self> {
        Aff {
            x: RFp2(
                mont_fp(
                    "10857046999023057135944570762232829481370756359578518086990519993285655852781",
                ),
                mont_fp(
                    "11559732032986387107991004021392285783925812861821192530917403151452391805634",
                ),
            ),
            y: RFp2(
                mont_fp(
                    "8495653923123431417604973247489272438418190587263600148770280649306958101930",
                ),
                mont_fp(
                    "4082367875863433681332203403145435568316851327593401208105741076214120093531",
                ),
            ),
            inf: false,
        }
    }
}

// ------------------------------------------------------------------------------------------------
// GPU harness
// ------------------------------------------------------------------------------------------------

/// Mirrors `Bn254TestParams` in bn254_test.metal.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Params {
    count: u32,
    op: u32,
    iters: u32,
    k: u32,
}

struct Gpu {
    device: Device,
    queue: CommandQueue,
    lib: Library,
}

impl Gpu {
    fn new() -> Option<Self> {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping");
            return None;
        };
        assert!(!crate::METAL_LIB.is_empty(), "groth16 metallib is empty");
        let lib = device.new_library_with_data(crate::METAL_LIB).unwrap();
        let queue = device.new_command_queue();
        Some(Self { device, queue, lib })
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

    fn out(&self, words: usize) -> Buffer {
        self.buf(&vec![0u32; words.max(1)])
    }

    fn read(buf: &Buffer, words: usize) -> Vec<u32> {
        let ptr = buf.contents() as *const u32;
        unsafe { std::slice::from_raw_parts(ptr, words) }.to_vec()
    }

    /// Dispatch `count` threads of `name`, buffers at 0.., params after them. Returns GPU seconds.
    fn run(&self, name: &str, bufs: &[&Buffer], params: Params) -> f64 {
        let f = self.lib.get_function(name, None).unwrap();
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&f)
            .unwrap();
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        for (i, b) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(b), 0);
        }
        enc.set_bytes(
            bufs.len() as u64,
            std::mem::size_of::<Params>() as u64,
            &params as *const Params as *const c_void,
        );
        let tg = pso.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_threads(
            MTLSize::new(params.count as u64, 1, 1),
            MTLSize::new(tg, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        assert_eq!(
            cb.status(),
            MTLCommandBufferStatus::Completed,
            "{name} failed"
        );
        gpu_seconds(cb)
    }
}

/// GPUEndTime - GPUStartTime of a completed command buffer (not wrapped by metal-rs 0.29).
// objc's `sel_impl!` checks `feature = "cargo-clippy"`, unknown to this crate's check-cfg.
#[allow(unexpected_cfgs)]
fn gpu_seconds(cb: &metal::CommandBufferRef) -> f64 {
    let (start, end): (f64, f64) =
        unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
    end - start
}

/// Collects mismatches; panics with the first few at the end.
struct Check {
    what: String,
    failures: usize,
    examples: Vec<String>,
}

impl Check {
    fn new(what: impl Into<String>) -> Self {
        Self {
            what: what.into(),
            failures: 0,
            examples: Vec::new(),
        }
    }
    fn eq<T: PartialEq + std::fmt::Debug>(
        &mut self,
        i: usize,
        got: T,
        want: T,
        input: &dyn Fn() -> String,
    ) {
        if got != want {
            self.failures += 1;
            if self.examples.len() < 5 {
                self.examples.push(format!(
                    "case {i}: input {}\n  got  {got:?}\n  want {want:?}",
                    input()
                ));
            }
        }
    }
    fn finish(self, cases: usize) {
        assert!(
            self.failures == 0,
            "{}: {} of {cases} cases differ:\n{}",
            self.what,
            self.failures,
            self.examples.join("\n")
        );
        println!("{}: {cases} cases ok", self.what);
    }
}

// ------------------------------------------------------------------------------------------------
// Field tests
// ------------------------------------------------------------------------------------------------

const OP_NAMES: [&str; 10] = [
    "add",
    "sub",
    "mul",
    "sqr",
    "neg",
    "dbl",
    "inv",
    "to_mont",
    "from_mont",
    "is_canonical",
];

/// Edge elements (Montgomery representations) for a prime field.
fn field_edges(f: &FieldRef) -> Vec<L> {
    let p = f.p;
    let one = big_from_u64(1);
    let pm1 = big_sub(&p, &one).0;
    let canonical: Vec<L> = vec![
        [0; 4],
        one,
        big_from_u64(2),
        big_from_u64(3),
        pm1,
        big_sub(&p, &big_from_u64(2)).0,
        big_shr1(&pm1),                   // (p - 1) / 2
        big_add(&big_shr1(&pm1), &one).0, // (p + 1) / 2
        big_from_u64(u32::MAX as u64),
        big_from_u64(1 << 32),
        big_from_u64(u64::MAX),
        pow2(64),
        pow2(128),
        f.reduce(&pow2(255)),
        f.reduce(&[u64::MAX; 4]), // (R - 1) mod p
        f.r,                      // R mod p
        f.r2,                     // R^2 mod p
        f.mont_reduce(&one),      // R^-1 mod p
    ];
    let mut out: Vec<L> = canonical.iter().map(|v| f.to_mont(v)).collect();
    // Raw limb patterns used directly as Montgomery representations.
    out.extend([
        [0; 4],
        one,
        pm1,
        big_sub(&p, &big_from_u64(2)).0,
        [u64::MAX, u64::MAX, u64::MAX, p[3] >> 1],
    ]);
    out
}

/// Raw 256-bit edge integers (including values >= p) for to_mont / from_mont / is_canonical.
fn raw_edges(f: &FieldRef) -> Vec<L> {
    let p = f.p;
    let one = big_from_u64(1);
    let mut v = vec![[0; 4], one, big_sub(&p, &one).0, p, big_add(&p, &one).0];
    let mut m = p;
    for _ in 1..5 {
        v.push(big_sub(&big_add(&m, &p).0, &one).0); // (k+1) p - 1
        m = big_add(&m, &p).0;
        v.push(m); // (k+1) p
    }
    v.extend([
        [u64::MAX; 4],                                   // R - 1
        big_sub(&[u64::MAX; 4], &one).0,                 // R - 2
        big_add(&big_sub(&[u64::MAX; 4], &p).0, &one).0, // R - p
        pow2(255),
        pow2(254),
        big_shl(&p, 1),
    ]);
    v
}

fn field_suite(gpu: &Gpu, kernel: &str, name: &str, f: &FieldRef, seed: u64) {
    let mut rng = Rng::new(seed);
    let edges = field_edges(f);
    let mut a = Vec::new();
    let mut b = Vec::new();
    for x in &edges {
        for y in &edges {
            a.push(*x);
            b.push(*y);
        }
    }
    for i in 0..N_RANDOM {
        let x = f.random(&mut rng);
        let y = match i % 1000 {
            0 => x,
            1 => f.neg(&x),
            _ => f.random(&mut rng),
        };
        a.push(x);
        b.push(y);
    }
    let n = a.len();
    let flat = |v: &[L]| {
        let mut w = Vec::with_capacity(v.len() * 8);
        v.iter().for_each(|x| words_of(x, &mut w));
        w
    };
    let (ba, bb, out) = (gpu.buf(&flat(&a)), gpu.buf(&flat(&b)), gpu.out(n * 8));
    for op in 0..7u32 {
        gpu.run(
            kernel,
            &[&out, &ba, &bb],
            Params {
                count: n as u32,
                op,
                ..Default::default()
            },
        );
        let got = Gpu::read(&out, n * 8);
        let mut check = Check::new(format!("{name} {}", OP_NAMES[op as usize]));
        for i in 0..n {
            let want = match op {
                0 => f.add(&a[i], &b[i]),
                1 => f.sub(&a[i], &b[i]),
                2 => f.mul(&a[i], &b[i]),
                3 => f.mul(&a[i], &a[i]),
                4 => f.neg(&a[i]),
                5 => f.add(&a[i], &a[i]),
                _ => f.inv(&a[i]),
            };
            check.eq(i, from_words(&got[i * 8..]), want, &|| {
                format!("{:x?} {:x?}", a[i], b[i])
            });
        }
        check.finish(n);
    }

    // Raw integers, including values >= p, for the conversions and the canonical check.
    let mut raw = raw_edges(f);
    raw.extend(edges.iter().copied());
    for i in 0..N_RANDOM {
        raw.push(if i % 2 == 0 {
            rng.big()
        } else {
            f.random(&mut rng)
        });
    }
    let n = raw.len();
    let (br, out) = (gpu.buf(&flat(&raw)), gpu.out(n * 8));
    for op in 7..10u32 {
        gpu.run(
            kernel,
            &[&out, &br, &br],
            Params {
                count: n as u32,
                op,
                ..Default::default()
            },
        );
        let got = Gpu::read(&out, n * 8);
        let mut check = Check::new(format!("{name} {}", OP_NAMES[op as usize]));
        for i in 0..n {
            let want = match op {
                7 => f.to_mont(&raw[i]),
                8 => f.mont_reduce(&raw[i]),
                _ => big_from_u64(!big_geq(&raw[i], &f.p) as u64),
            };
            check.eq(i, from_words(&got[i * 8..]), want, &|| {
                format!("{:x?}", raw[i])
            });
        }
        check.finish(n);
    }
}

#[test]
fn reference_self_check() {
    let mut rng = Rng::new(1);
    for (f, p_limbs) in [
        (
            fp(),
            [
                0x3c208c16d87cfd47,
                0x97816a916871ca8d,
                0xb85045b68181585d,
                0x30644e72e131a029,
            ],
        ),
        (
            fr(),
            [
                0x43e1f593f0000001,
                0x2833e84879b97091,
                0xb85045b68181585d,
                0x30644e72e131a029,
            ],
        ),
    ] {
        assert_eq!(f.p, p_limbs);
        // Montgomery mul against double-and-add: mont(a, b) * R == a * b (mod p).
        for i in 0..2000 {
            let a = if i < 4 {
                big_sub(&f.p, &big_from_u64(i + 1)).0
            } else {
                f.random(&mut rng)
            };
            let b = f.random(&mut rng);
            assert_eq!(f.slow_mulmod(&f.mul(&a, &b), &f.r), f.slow_mulmod(&a, &b));
            assert_eq!(
                f.mul(&f.to_mont(&a), &f.inv(&f.to_mont(&a))),
                if a == [0; 4] { [0; 4] } else { f.r }
            );
        }
        assert_eq!(f.slow_mulmod(&f.r, &f.r), f.r2);
    }
    // Fp2 inverse and the curve generators.
    for _ in 0..100 {
        let a = RFp2::random(&mut rng);
        assert_eq!(a.mul(&a.inv()), RFp2::one());
    }
    assert!(RFp::generator().on_curve(&RFp::b()));
    assert!(RFp2::generator().on_curve(&RFp2::b()));
    // Reference group law sanity: (k1 + k2) G == k1 G + k2 G, r G == O for G1.
    let g = RFp::generator();
    let ks = [12345u64, 678910, 12345 + 678910];
    let m = batch_mul(&[g; 3], &ks, 64);
    assert_eq!(batch_add(&m[..1], &m[1..2])[0], m[2]);
    let mut acc = Aff::inf();
    let mut base = g;
    for i in 0..256 {
        if big_bit(&fr().p, i) {
            acc = batch_add(&[acc], &[base])[0];
        }
        base = batch_add(&[base], &[base])[0];
    }
    assert!(acc.inf, "r * G1 != O");
}

#[test]
fn fp_ops() {
    let Some(gpu) = Gpu::new() else { return };
    field_suite(&gpu, "bn254_test_fp_op", "Fp", fp(), 0xf9);
}

#[test]
fn fr_ops() {
    let Some(gpu) = Gpu::new() else { return };
    field_suite(&gpu, "bn254_test_fr_op", "Fr", fr(), 0xf7);
}

#[test]
fn fp2_ops() {
    let Some(gpu) = Gpu::new() else { return };
    let mut rng = Rng::new(0xf2);
    let f = fp();
    let pm1 = big_sub(&f.p, &big_from_u64(1)).0;
    let small = [
        RFp::zero(),
        RFp::one(),
        RFp::one().neg(),
        RFp(big_from_u64(1)),
        RFp(pm1),
    ];
    let mut edges = Vec::new();
    for x in &small {
        for y in &small {
            edges.push(RFp2(*x, *y));
        }
    }
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for x in &edges {
        for y in &edges {
            a.push(*x);
            b.push(*y);
        }
    }
    for i in 0..N_RANDOM {
        let x = RFp2::random(&mut rng);
        let y = match i % 1000 {
            0 => x,
            1 => x.neg(),
            _ => RFp2::random(&mut rng),
        };
        a.push(x);
        b.push(y);
    }
    let n = a.len();
    let flat = |v: &[RFp2]| {
        let mut w = Vec::with_capacity(v.len() * 16);
        v.iter().for_each(|x| x.write(&mut w));
        w
    };
    let (ba, bb, out) = (gpu.buf(&flat(&a)), gpu.buf(&flat(&b)), gpu.out(n * 16));
    for op in 0..7u32 {
        gpu.run(
            "bn254_test_fp2_op",
            &[&out, &ba, &bb],
            Params {
                count: n as u32,
                op,
                ..Default::default()
            },
        );
        let got = Gpu::read(&out, n * 16);
        let mut check = Check::new(format!("Fp2 {}", OP_NAMES[op as usize]));
        for i in 0..n {
            let want = match op {
                0 => a[i].add(&b[i]),
                1 => a[i].sub(&b[i]),
                2 => a[i].mul(&b[i]),
                3 => a[i].mul(&a[i]),
                4 => a[i].neg(),
                5 => a[i].add(&a[i]),
                _ => a[i].inv(),
            };
            let g = RFp2::read(&got[i * 16..]);
            check.eq(i, g, want, &|| format!("{:x?} {:x?}", a[i], b[i]));
            if op == 6 && !a[i].is_zero() {
                check.eq(i, g.mul(&a[i]), RFp2::one(), &|| {
                    format!("inverse of {:x?}", a[i])
                });
            }
        }
        check.finish(n);
    }
}

// ------------------------------------------------------------------------------------------------
// Curve tests
// ------------------------------------------------------------------------------------------------

const CURVE_OP_NAMES: [&str; 10] = [
    "add_affine",
    "add",
    "dbl",
    "dbl_affine",
    "neg",
    "from_affine",
    "from_jacobian",
    "to_affine",
    "to_jacobian",
    "mul_small",
];

fn random_nonzero<F: RF>(rng: &mut Rng) -> F {
    loop {
        let v = F::random(rng);
        if !v.is_zero() {
            return v;
        }
    }
}

fn curve_suite<F: Curve>(seed: u64) {
    let Some(gpu) = Gpu::new() else { return };
    let mut rng = Rng::new(seed);
    let bcoef = F::b();
    let g = F::generator();
    assert!(g.on_curve(&bcoef));

    // Random points: 1024 random 64-bit multiples of the generator, then pairwise sums.
    let nb = 1024;
    let ks: Vec<u64> = (0..nb).map(|_| rng.next()).collect();
    let base = batch_mul(&vec![g; nb], &ks, 64);
    let rand_pts = |count: usize, rng: &mut Rng| {
        let i: Vec<_> = (0..count)
            .map(|_| base[(rng.next() % nb as u64) as usize])
            .collect();
        let j: Vec<_> = (0..count)
            .map(|_| base[(rng.next() % nb as u64) as usize])
            .collect();
        batch_add(&i, &j)
    };
    let n_edge = 16;
    let n = n_edge + N_RANDOM;
    let mut pa = rand_pts(n, &mut rng);
    let mut pb = rand_pts(n, &mut rng);
    let mut pc = rand_pts(n, &mut rng);
    let mut pd = rand_pts(n, &mut rng);
    let mut la: Vec<F> = (0..n).map(|_| random_nonzero(&mut rng)).collect();
    let mut lc: Vec<F> = (0..n).map(|_| random_nonzero(&mut rng)).collect();
    let mut zd: Vec<F> = (0..n).map(|_| random_nonzero(&mut rng)).collect();
    let mut kk: Vec<u32> = (0..n).map(|_| rng.next() as u32).collect();
    let inf = Aff::<F>::inf();

    // Edge cases (indices 0..n_edge).
    pa[0] = inf;
    pb[0] = inf;
    pc[0] = inf;
    pd[0] = inf; // inf + inf
    pa[1] = inf; // inf + Q, inf + C
    pb[2] = inf;
    pc[2] = inf;
    pd[2] = inf; // P + inf
    pb[3] = pa[3];
    pc[3] = pa[3]; // P + P (doubling branch, different scales)
    pb[4] = pa[4].neg();
    pc[4] = pa[4].neg(); // P + (-P)
    pb[5] = pa[5];
    pc[5] = pa[5];
    lc[5] = la[5]; // P + P, identical representation
    pb[6] = pa[6].neg();
    pc[6] = pa[6].neg();
    lc[6] = la[6]; // P + (-P), same scale
    pb[7] = g;
    pa[7] = g;
    pc[7] = g.neg(); // generator
    zd[8] = F::one(); // Jacobian with Z = 1
    kk[9] = 0;
    kk[10] = 1;
    kk[11] = 2;
    kk[12] = u32::MAX;
    kk[13] = 1 << 31;
    pa[14] = inf;
    kk[14] = 7; // k * inf
                // 15: purely random.
    la[8] = F::one(); // Xyzz with ZZ = ZZZ = 1

    for p in pa.iter().chain(&pb).chain(&pc).chain(&pd) {
        assert!(p.on_curve(&bcoef));
    }

    let (mut wa, mut wb, mut wc, mut wd) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        write_xyzz(&pa[i], &la[i], &mut wa);
        pb[i].write(&mut wb);
        write_xyzz(&pc[i], &lc[i], &mut wc);
        write_jacobian(&pd[i], &zd[i], &mut wd);
    }
    // Index 1: infinity with junk X, Y (only ZZ = 0 marks infinity).
    {
        let w = F::WORDS;
        let junk: Vec<u32> = {
            let mut v = Vec::new();
            F::random(&mut rng).write(&mut v);
            F::random(&mut rng).write(&mut v);
            v
        };
        wa[4 * w..6 * w].copy_from_slice(&junk);
    }
    let fw = F::WORDS;
    let bufs = [
        gpu.out(n * 4 * fw),
        gpu.out(n * 2 * fw),
        gpu.out(n * 3 * fw),
        gpu.buf(&wa),
        gpu.buf(&wb),
        gpu.buf(&wc),
        gpu.buf(&wd),
        gpu.buf(&kk),
    ];
    let refs: Vec<&Buffer> = bufs.iter().collect();

    for op in 0..10u32 {
        let secs = gpu.run(
            F::KERNEL,
            &refs,
            Params {
                count: n as u32,
                op,
                ..Default::default()
            },
        );
        let got: Vec<Result<Aff<F>, String>> = match op {
            7 => read_affine(&Gpu::read(&bufs[1], n * 2 * fw))
                .into_iter()
                .map(Ok)
                .collect(),
            8 => read_jacobian(&Gpu::read(&bufs[2], n * 3 * fw))
                .into_iter()
                .map(Ok)
                .collect(),
            _ => read_xyzz(&Gpu::read(&bufs[0], n * 4 * fw)),
        };
        let want: Vec<Aff<F>> = match op {
            0 => batch_add(&pa, &pb),
            1 => batch_add(&pa, &pc),
            2 => batch_add(&pa, &pa),
            3 => batch_add(&pb, &pb),
            4 => pa.iter().map(|p| p.neg()).collect(),
            5 => pb.clone(),
            6 => pd.clone(),
            7 | 8 => pa.clone(),
            _ => batch_mul(&pa, &kk.iter().map(|&k| k as u64).collect::<Vec<_>>(), 32),
        };
        let mut check = Check::new(format!("{} {}", F::NAME, CURVE_OP_NAMES[op as usize]));
        for i in 0..n {
            check.eq(i, got[i].clone(), Ok(want[i]), &|| {
                format!(
                    "a={:?} b={:?} c={:?} d={:?} k={}",
                    pa[i], pb[i], pc[i], pd[i], kk[i]
                )
            });
        }
        if op == 7 {
            // to_affine must also be bit-exact on the words (infinity = (0, 0)).
            let words = Gpu::read(&bufs[1], n * 2 * fw);
            for i in 0..n {
                let mut w = Vec::new();
                want[i].write(&mut w);
                check.eq(i, words[i * 2 * fw..(i + 1) * 2 * fw].to_vec(), w, &|| {
                    format!("{:?}", pa[i])
                });
            }
        }
        check.finish(n);
        println!("  ({:.2} ms GPU)", secs * 1e3);
    }
}

#[test]
fn g1_ops() {
    curve_suite::<RFp>(0x61);
}

#[test]
fn g2_ops() {
    curve_suite::<RFp2>(0x62);
}

// ------------------------------------------------------------------------------------------------
// Throughput (indicative only; other GPU work may be running)
// ------------------------------------------------------------------------------------------------

#[test]
fn throughput() {
    let Some(gpu) = Gpu::new() else { return };
    let f = fp();
    let mut rng = Rng::new(0xbe);

    // Fp mul: 2^20 threads x 512 dependent multiplications.
    let count = 1usize << 20;
    let iters = 512u32;
    let input: Vec<L> = (0..count).map(|_| f.random(&mut rng)).collect();
    let mut w = Vec::with_capacity(count * 8);
    input.iter().for_each(|x| words_of(x, &mut w));
    let (bin, out) = (gpu.buf(&w), gpu.out(count * 8));
    let p = Params {
        count: count as u32,
        iters,
        ..Default::default()
    };
    let best = (0..3)
        .map(|_| gpu.run("bn254_bench_fp_mul", &[&out, &bin], p))
        .fold(f64::MAX, f64::min);
    let got = Gpu::read(&out, count * 8);
    for i in 0..64 {
        let y = input[(i + 1) % count];
        let mut x = input[i];
        for _ in 0..iters {
            x = f.mul(&x, &y);
        }
        assert_eq!(from_words(&got[i * 8..]), x, "bench_fp_mul thread {i}");
    }
    let rate = (count as f64) * iters as f64 / best;
    println!(
        "Fp mul: {:.3} Gop/s ({count} x {iters} in {:.2} ms GPU)",
        rate / 1e9,
        best * 1e3
    );

    // G1 mixed addition: 2^18 threads x 256 additions over 4096 random points.
    let np = 4096;
    let g = RFp::generator();
    let ks: Vec<u64> = (0..np).map(|_| rng.next()).collect();
    let pts = batch_mul(&vec![g; np], &ks, 64);
    let mut w = Vec::new();
    pts.iter().for_each(|p| p.write(&mut w));
    let count = 1usize << 18;
    let iters = 256u32;
    let (bpts, out) = (gpu.buf(&w), gpu.out(count * 32));
    let p = Params {
        count: count as u32,
        iters,
        k: np as u32,
        ..Default::default()
    };
    let best = (0..3)
        .map(|_| gpu.run("bn254_bench_g1_add_affine", &[&out, &bpts], p))
        .fold(f64::MAX, f64::min);
    let got = read_xyzz::<RFp>(&Gpu::read(&out, count * 32));
    let check_n = 16;
    let mut acc: Vec<Aff<RFp>> = (0..check_n).map(|i| pts[i % np]).collect();
    for j in 1..iters as usize {
        let q: Vec<_> = (0..check_n).map(|i| pts[(i + j) % np]).collect();
        acc = batch_add(&acc, &q);
    }
    for i in 0..check_n {
        assert_eq!(got[i], Ok(acc[i]), "bench_g1_add_affine thread {i}");
    }
    let rate = (count as f64) * (iters - 1) as f64 / best;
    println!(
        "G1 add_affine (XYZZ += affine): {:.1} Mop/s ({count} x {} in {:.2} ms GPU)",
        rate / 1e6,
        iters - 1,
        best * 1e3
    );
}
