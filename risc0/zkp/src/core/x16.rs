// Copyright 2025 RISC Zero, Inc.
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

//! Baby Bear arithmetic 16 elements at a time (AVX-512, or AVX2 as two halves) for the CPU prover:
//! - Poseidon2 for its Merkle trees, 16 hashes at once: the rows of a column-major matrix
//!   ([`hash_rows`]) and pairs of digests ([`hash_pairs`]); bit-identical to
//!   [`super::hash::poseidon2::unpadded_hash`], each lane one hash;
//! - the NTT ([`evaluate_ntt`], [`interpolate_ntt`]): the same butterflies as [`super::ntt`],
//!   16 at a time where a level has that many, so the results are identical too.
//!
//! Elements stay in risc0's Montgomery form (R = 2^32), always reduced to [0, P).

use risc0_core::field::{Elem as _, RootsOfUnity as _};

use super::hash::poseidon2::{
    CELLS, CELLS_OUT, CELLS_RATE, M_INT_DIAG_HZN, ROUNDS_HALF_FULL, ROUNDS_PARTIAL, ROUND_CONSTANTS,
};
use crate::core::digest::{Digest, DIGEST_WORDS};
use risc0_core::field::baby_bear::BabyBearElem;

/// Hashes per call.
pub const LANES: usize = 16;

const P: u32 = 15 * (1 << 27) + 1;
/// P^-1 mod 2^32.
const M: u32 = 0x88000001;

#[derive(Clone, Copy)]
enum Isa {
    Avx512,
    Avx2,
}

fn isa() -> Option<Isa> {
    static ISA: std::sync::OnceLock<Option<Isa>> = std::sync::OnceLock::new();
    *ISA.get_or_init(|| {
        if std::env::var_os("RISC0_CPU_HASH_SCALAR").is_some() {
            return None;
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            Some(Isa::Avx512)
        } else if std::arch::is_x86_feature_detected!("avx2") {
            Some(Isa::Avx2)
        } else {
            None
        }
    })
}

/// Whether this CPU has the 16-wide arithmetic.
pub fn available() -> bool {
    isa().is_some()
}

/// The digests of rows `first..first + 16` of `matrix`, which holds `cols` columns of `rows`
/// elements each (element (row, col) at `col * rows + row`).
pub fn hash_rows(
    matrix: &[BabyBearElem],
    rows: usize,
    cols: usize,
    first: usize,
) -> [Digest; LANES] {
    assert!(first + LANES <= rows && matrix.len() >= cols * rows);
    // SAFETY: the CPU has the instruction set (`isa`); the reads are in bounds (asserted).
    unsafe {
        match isa().expect("hash_rows: no 16-wide Poseidon2 on this CPU") {
            Isa::Avx512 => avx512::hash_rows(matrix, rows, cols, first),
            Isa::Avx2 => avx2::hash_rows(matrix, rows, cols, first),
        }
    }
}

/// `hash_pair(input[2i], input[2i + 1])` for i in 0..16.
pub fn hash_pairs(input: &[Digest]) -> [Digest; LANES] {
    assert_eq!(input.len(), 2 * LANES);
    // SAFETY: the CPU has the instruction set (`isa`).
    unsafe {
        match isa().expect("hash_pairs: no 16-wide Poseidon2 on this CPU") {
            Isa::Avx512 => avx512::hash_pairs(input),
            Isa::Avx2 => avx2::hash_pairs(input),
        }
    }
}

/// The forward NTT of `io` (a power of two long), as [`super::ntt::evaluate_ntt`].
pub fn evaluate_ntt(io: &mut [BabyBearElem], expand_bits: usize) {
    let n = super::log2_ceil(io.len());
    assert_eq!(1 << n, io.len());
    // SAFETY: the CPU has the instruction set (`isa`); `io` holds 2^n elements.
    unsafe {
        match isa().expect("evaluate_ntt: no 16-wide arithmetic on this CPU") {
            Isa::Avx512 => avx512::fwd(io.as_mut_ptr() as *mut u32, n, expand_bits),
            Isa::Avx2 => avx2::fwd(io.as_mut_ptr() as *mut u32, n, expand_bits),
        }
    }
}

/// The inverse NTT of `io` (a power of two long), as [`super::ntt::interpolate_ntt`].
pub fn interpolate_ntt(io: &mut [BabyBearElem]) {
    let n = super::log2_ceil(io.len());
    assert_eq!(1 << n, io.len());
    // SAFETY: as in `evaluate_ntt`.
    unsafe {
        match isa().expect("interpolate_ntt: no 16-wide arithmetic on this CPU") {
            Isa::Avx512 => avx512::rev(io.as_mut_ptr() as *mut u32, n),
            Isa::Avx2 => avx2::rev(io.as_mut_ptr() as *mut u32, n),
        }
    }
    let norm = BabyBearElem::from_u64(io.len() as u64).inv();
    for x in io.iter_mut() {
        *x = *x * norm;
    }
}

/// Levels below this many points are left to the scalar butterflies of `super::ntt`.
const NTT_MIN_PO2: usize = 5;

fn raw(x: BabyBearElem) -> u32 {
    x.as_u32_montgomery()
}

/// The permutation and the two hashes, over a 16-lane vector type `V` with `splat`, `add`, `sub`,
/// `mul`, `load`, `store`, all `#[target_feature(enable = $feat)]`.
macro_rules! poseidon2_x16 {
    ($feat:literal) => {
        #[target_feature(enable = $feat)]
        fn sbox(x: V) -> V {
            let x2 = mul(x, x);
            let x4 = mul(x2, x2);
            let x6 = mul(x4, x2);
            mul(x6, x)
        }

        #[target_feature(enable = $feat)]
        fn m_ext(cells: &mut [V; CELLS]) {
            // As `multiply_by_m_ext`: the 4x4 circulant on each group of four, plus the sums over
            // the groups.
            let mut sums = [splat(0); 4];
            for g in 0..CELLS / 4 {
                let x = &mut cells[g * 4..g * 4 + 4];
                let t0 = add(x[0], x[1]);
                let t1 = add(x[2], x[3]);
                let t2 = add(add(x[1], x[1]), t1);
                let t3 = add(add(x[3], x[3]), t0);
                let t1_2 = add(t1, t1);
                let t4 = add(add(t1_2, t1_2), t3);
                let t0_2 = add(t0, t0);
                let t5 = add(add(t0_2, t0_2), t2);
                let t6 = add(t3, t5);
                let t7 = add(t2, t4);
                x[0] = t6;
                x[1] = t5;
                x[2] = t7;
                x[3] = t4;
                for j in 0..4 {
                    sums[j] = add(sums[j], x[j]);
                }
            }
            for i in 0..CELLS {
                cells[i] = add(cells[i], sums[i % 4]);
            }
        }

        #[target_feature(enable = $feat)]
        fn m_int(cells: &mut [V; CELLS]) {
            let mut sum = cells[0];
            for c in &cells[1..] {
                sum = add(sum, *c);
            }
            for i in 0..CELLS {
                cells[i] = add(sum, mul(splat(raw(M_INT_DIAG_HZN[i])), cells[i]));
            }
        }

        #[target_feature(enable = $feat)]
        fn mix(cells: &mut [V; CELLS]) {
            let mut round = 0;
            m_ext(cells);
            for _ in 0..ROUNDS_HALF_FULL {
                for i in 0..CELLS {
                    cells[i] = sbox(add(
                        cells[i],
                        splat(raw(ROUND_CONSTANTS[round * CELLS + i])),
                    ));
                }
                m_ext(cells);
                round += 1;
            }
            for _ in 0..ROUNDS_PARTIAL {
                cells[0] = sbox(add(cells[0], splat(raw(ROUND_CONSTANTS[round * CELLS]))));
                m_int(cells);
                round += 1;
            }
            for _ in 0..ROUNDS_HALF_FULL {
                for i in 0..CELLS {
                    cells[i] = sbox(add(
                        cells[i],
                        splat(raw(ROUND_CONSTANTS[round * CELLS + i])),
                    ));
                }
                m_ext(cells);
                round += 1;
            }
        }

        #[target_feature(enable = $feat)]
        fn digests(cells: &[V; CELLS]) -> [Digest; LANES] {
            let mut words = [[0u32; LANES]; CELLS_OUT];
            for (w, c) in words.iter_mut().zip(cells) {
                // SAFETY: `w` holds LANES words.
                unsafe { store(w.as_mut_ptr(), *c) };
            }
            std::array::from_fn(|lane| {
                let d: [u32; DIGEST_WORDS] = std::array::from_fn(|i| words[i][lane]);
                Digest::from(d)
            })
        }

        /// The forward butterflies of `super::ntt` on the 2^n elements at `io`.
        /// SAFETY: needs the instruction set; `io` holds 2^n elements.
        #[target_feature(enable = $feat)]
        pub(super) unsafe fn fwd(io: *mut u32, n: usize, expand_bits: usize) {
            if n == expand_bits {
                return;
            }
            if n < NTT_MIN_PO2 {
                let io = unsafe { std::slice::from_raw_parts_mut(io as *mut BabyBearElem, 1 << n) };
                crate::core::ntt::fwd_butterfly_small::<BabyBearElem, BabyBearElem>(
                    io,
                    n,
                    expand_bits,
                );
                return;
            }
            let half = 1 << (n - 1);
            unsafe {
                fwd(io, n - 1, expand_bits);
                fwd(io.add(half), n - 1, expand_bits);
            }
            let (mut cur, step) = twiddles(BabyBearElem::ROU_FWD[n]);
            for i in (0..half).step_by(LANES) {
                unsafe {
                    let a = load(io.add(i));
                    let b = mul(load(io.add(i + half)), cur);
                    store(io.add(i), add(a, b));
                    store(io.add(i + half), sub(a, b));
                }
                cur = mul(cur, step);
            }
        }

        /// The inverse butterflies of `super::ntt` (without the normalization).
        /// SAFETY: needs the instruction set; `io` holds 2^n elements.
        #[target_feature(enable = $feat)]
        pub(super) unsafe fn rev(io: *mut u32, n: usize) {
            if n < NTT_MIN_PO2 {
                let io = unsafe { std::slice::from_raw_parts_mut(io as *mut BabyBearElem, 1 << n) };
                crate::core::ntt::rev_butterfly_small::<BabyBearElem, BabyBearElem>(io, n);
                return;
            }
            let half = 1 << (n - 1);
            let (mut cur, step) = twiddles(BabyBearElem::ROU_REV[n]);
            for i in (0..half).step_by(LANES) {
                unsafe {
                    let a = load(io.add(i));
                    let b = load(io.add(i + half));
                    store(io.add(i), add(a, b));
                    store(io.add(i + half), mul(sub(a, b), cur));
                }
                cur = mul(cur, step);
            }
            unsafe {
                rev(io, n - 1);
                rev(io.add(half), n - 1);
            }
        }

        /// (w^0, .., w^15) and w^16.
        #[target_feature(enable = $feat)]
        fn twiddles(w: BabyBearElem) -> (V, V) {
            let mut lanes = [0u32; LANES];
            let mut c = BabyBearElem::ONE;
            for l in &mut lanes {
                *l = raw(c);
                c *= w;
            }
            (unsafe { load(lanes.as_ptr()) }, splat(raw(c)))
        }

        /// SAFETY: needs the instruction set; `first + LANES <= rows`, `matrix.len() >= cols * rows`.
        #[target_feature(enable = $feat)]
        pub(super) unsafe fn hash_rows(
            matrix: &[BabyBearElem],
            rows: usize,
            cols: usize,
            first: usize,
        ) -> [Digest; LANES] {
            let base = matrix.as_ptr() as *const u32;
            let mut cells = [splat(0); CELLS];
            let mut unmixed = 0;
            for col in 0..cols {
                cells[unmixed] = unsafe { load(base.add(col * rows + first)) };
                unmixed += 1;
                if unmixed == CELLS_RATE {
                    mix(&mut cells);
                    unmixed = 0;
                }
            }
            if unmixed != 0 || cols == 0 {
                for c in &mut cells[unmixed..CELLS_RATE] {
                    *c = splat(0);
                }
                mix(&mut cells);
            }
            digests(&cells)
        }

        /// SAFETY: needs the instruction set.
        #[target_feature(enable = $feat)]
        pub(super) unsafe fn hash_pairs(input: &[Digest]) -> [Digest; LANES] {
            // Cell i of lane l: word i of the left digest (i < 8) or word i - 8 of the right one.
            let mut words = [[0u32; LANES]; 2 * DIGEST_WORDS];
            for lane in 0..LANES {
                let (l, r) = (&input[2 * lane], &input[2 * lane + 1]);
                for i in 0..DIGEST_WORDS {
                    words[i][lane] = l.as_words()[i];
                    words[DIGEST_WORDS + i][lane] = r.as_words()[i];
                }
            }
            let mut cells = [splat(0); CELLS];
            for (c, w) in cells.iter_mut().zip(&words) {
                *c = unsafe { load(w.as_ptr()) };
            }
            // 16 inputs: exactly CELLS_RATE, one mix, no padding.
            const { assert!(2 * DIGEST_WORDS == CELLS_RATE) };
            mix(&mut cells);
            digests(&cells)
        }
    };
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;
    use std::arch::x86_64::*;

    #[derive(Clone, Copy)]
    struct V(__m512i);

    #[target_feature(enable = "avx512f")]
    fn splat(x: u32) -> V {
        V(_mm512_set1_epi32(x as i32))
    }
    #[target_feature(enable = "avx512f")]
    unsafe fn load(p: *const u32) -> V {
        V(unsafe { _mm512_loadu_si512(p as *const _) })
    }
    #[target_feature(enable = "avx512f")]
    unsafe fn store(p: *mut u32, v: V) {
        unsafe { _mm512_storeu_si512(p as *mut _, v.0) }
    }
    #[target_feature(enable = "avx512f")]
    fn add(a: V, b: V) -> V {
        let r = _mm512_add_epi32(a.0, b.0);
        V(_mm512_min_epu32(
            r,
            _mm512_sub_epi32(r, _mm512_set1_epi32(P as i32)),
        ))
    }
    #[allow(dead_code)]
    #[target_feature(enable = "avx512f")]
    fn sub(a: V, b: V) -> V {
        let r = _mm512_sub_epi32(a.0, b.0);
        V(_mm512_min_epu32(
            r,
            _mm512_add_epi32(r, _mm512_set1_epi32(P as i32)),
        ))
    }
    /// Montgomery: (a*b - m*P) / 2^32 with m = lo(a*b) * P^-1 lies in (-P, P).
    #[target_feature(enable = "avx512f")]
    fn mul(a: V, b: V) -> V {
        let p = _mm512_set1_epi32(P as i32);
        let m = _mm512_set1_epi32(M as i32);
        let te = _mm512_mul_epu32(a.0, b.0);
        let to = _mm512_mul_epu32(_mm512_srli_epi64(a.0, 32), _mm512_srli_epi64(b.0, 32));
        let qe = _mm512_mul_epu32(_mm512_mul_epu32(te, m), p);
        let qo = _mm512_mul_epu32(_mm512_mul_epu32(to, m), p);
        let de = _mm512_srli_epi64(_mm512_sub_epi64(te, qe), 32);
        let dn = _mm512_sub_epi64(to, qo);
        let r = _mm512_mask_blend_epi32(0xAAAA, de, dn);
        V(_mm512_min_epu32(r, _mm512_add_epi32(r, p)))
    }

    poseidon2_x16!("avx512f");
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    #[derive(Clone, Copy)]
    struct V(__m256i, __m256i);

    #[target_feature(enable = "avx2")]
    fn splat(x: u32) -> V {
        let a = _mm256_set1_epi32(x as i32);
        V(a, a)
    }
    #[target_feature(enable = "avx2")]
    unsafe fn load(p: *const u32) -> V {
        unsafe {
            V(
                _mm256_loadu_si256(p as *const __m256i),
                _mm256_loadu_si256(p.add(8) as *const __m256i),
            )
        }
    }
    #[target_feature(enable = "avx2")]
    unsafe fn store(p: *mut u32, v: V) {
        unsafe {
            _mm256_storeu_si256(p as *mut __m256i, v.0);
            _mm256_storeu_si256(p.add(8) as *mut __m256i, v.1);
        }
    }
    #[target_feature(enable = "avx2")]
    fn add8(a: __m256i, b: __m256i) -> __m256i {
        let r = _mm256_add_epi32(a, b);
        _mm256_min_epu32(r, _mm256_sub_epi32(r, _mm256_set1_epi32(P as i32)))
    }
    #[target_feature(enable = "avx2")]
    fn sub8(a: __m256i, b: __m256i) -> __m256i {
        let r = _mm256_sub_epi32(a, b);
        _mm256_min_epu32(r, _mm256_add_epi32(r, _mm256_set1_epi32(P as i32)))
    }
    #[target_feature(enable = "avx2")]
    fn mul8(a: __m256i, b: __m256i) -> __m256i {
        let p = _mm256_set1_epi32(P as i32);
        let m = _mm256_set1_epi32(M as i32);
        let te = _mm256_mul_epu32(a, b);
        let to = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b, 32));
        let qe = _mm256_mul_epu32(_mm256_mul_epu32(te, m), p);
        let qo = _mm256_mul_epu32(_mm256_mul_epu32(to, m), p);
        let de = _mm256_srli_epi64(_mm256_sub_epi64(te, qe), 32);
        let dn = _mm256_sub_epi64(to, qo);
        let r = _mm256_blend_epi32(de, dn, 0xAA);
        _mm256_min_epu32(r, _mm256_add_epi32(r, p))
    }
    #[target_feature(enable = "avx2")]
    fn add(a: V, b: V) -> V {
        V(add8(a.0, b.0), add8(a.1, b.1))
    }
    #[allow(dead_code)]
    #[target_feature(enable = "avx2")]
    fn sub(a: V, b: V) -> V {
        V(sub8(a.0, b.0), sub8(a.1, b.1))
    }
    #[target_feature(enable = "avx2")]
    fn mul(a: V, b: V) -> V {
        V(mul8(a.0, b.0), mul8(a.1, b.1))
    }

    poseidon2_x16!("avx2");
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::core::hash::poseidon2::Poseidon2HashSuite;

    #[test]
    fn ntt_matches_scalar() {
        if !available() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        for n in 0..=12 {
            for expand_bits in [0, 2] {
                let input: Vec<BabyBearElem> = (0..1usize << n)
                    .map(|_| BabyBearElem::new(rng.random::<u32>() % P))
                    .collect();
                let mut want = input.clone();
                crate::core::ntt::evaluate_ntt::<BabyBearElem, BabyBearElem>(
                    &mut want,
                    expand_bits,
                );
                let mut got = input.clone();
                evaluate_ntt(&mut got, expand_bits);
                assert_eq!(got, want, "fwd n {n} expand {expand_bits}");
                if std::arch::is_x86_feature_detected!("avx2") {
                    let mut got = input.clone();
                    unsafe { avx2::fwd(got.as_mut_ptr() as *mut u32, n, expand_bits) };
                    assert_eq!(got, want, "avx2 fwd n {n}");
                }
            }
            let input: Vec<BabyBearElem> = (0..1usize << n)
                .map(|_| BabyBearElem::new(rng.random::<u32>() % P))
                .collect();
            let mut want = input.clone();
            crate::core::ntt::interpolate_ntt::<BabyBearElem, BabyBearElem>(&mut want);
            let mut got = input;
            interpolate_ntt(&mut got);
            assert_eq!(got, want, "rev n {n}");
        }
    }

    #[test]
    fn matches_scalar() {
        if !available() {
            return;
        }
        let suite = Poseidon2HashSuite::new_suite();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for cols in [0, 1, 15, 16, 17, 32, 211] {
            let rows = 64;
            let matrix: Vec<BabyBearElem> = (0..rows * cols)
                .map(|_| BabyBearElem::new(rng.random::<u32>() % P))
                .collect();
            for first in [0, 16, 48] {
                let got = hash_rows(&matrix, rows, cols, first);
                if std::arch::is_x86_feature_detected!("avx2") {
                    assert_eq!(got, unsafe { avx2::hash_rows(&matrix, rows, cols, first) });
                }
                for lane in 0..LANES {
                    let column: Vec<_> =
                        (0..cols).map(|c| matrix[c * rows + first + lane]).collect();
                    assert_eq!(
                        got[lane],
                        *suite.hashfn.hash_elem_slice(&column),
                        "cols {cols}"
                    );
                }
            }
        }
        let digests: Vec<Digest> = (0..2 * LANES)
            .map(|_| Digest::from(std::array::from_fn(|_| rng.random::<u32>() % P)))
            .collect();
        let got = hash_pairs(&digests);
        if std::arch::is_x86_feature_detected!("avx2") {
            assert_eq!(got, unsafe { avx2::hash_pairs(&digests) });
        }
        for lane in 0..LANES {
            assert_eq!(
                got[lane],
                *suite
                    .hashfn
                    .hash_pair(&digests[2 * lane], &digests[2 * lane + 1])
            );
        }
    }
}
