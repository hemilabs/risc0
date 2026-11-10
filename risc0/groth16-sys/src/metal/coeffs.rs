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

//! The preprocessed QAP coefficients (`preprocessed_coeffs.bin`) in shared Metal buffers and the
//! fuzzed MSM results (`fuzzed_msm_results.bin`) of the rzup `risc0-groth16` component
//! (docs/metal-port/PLAN.md 4.5, item G3).
//!
//! Ports of the CUDA prover's `preprocessed_coeffs(gpu, file_path)` constructor
//! (`kernels/cuda/groth16_coeffs.cuh`, non-Windows, non-compact path) and
//! `groth16_prover::read_fuzzed_results_from_file` (`kernels/cuda/groth16_prover.cuh`), with the
//! Intel prover's `Coeffs`/`read_fuzzed_results` (`kernels/intel/groth16_intel.hpp`) as the
//! second reference.
//!
//! Coefficients file: 4 x u64 header (coefficient counts of A and B, index counts of A and B),
//! the coefficients (sppark `coeff_t` = {u32 m, c, s; fr_t value}, 48 bytes with the value
//! 8-aligned), then the row start indices of A followed by those of B (each list ends with its
//! coefficient count). Row `i` of A is coefficients `[ind_a[i], ind_a[i + 1])`; they all share the
//! output index `c` and `witness_into_poly` writes `out[c] = sum witness[s] * value`. Values are
//! Fr in Montgomery form.

use std::{fs::File, mem::size_of, path::Path, time::Instant};

use anyhow::{bail, ensure, Context, Result};
use ark_bn254::{Fr, G1Projective, G2Projective};
use ark_ff::{BigInt, PrimeField};
use metal::DeviceRef;

use super::srs::{
    limbs_le, read_exact_at_par, G1JacobianRaw, G2JacobianRaw, LoadStats, Pod, SharedBuf,
};

/// sppark `coeff_t` as stored in the preprocessed file (and read by the kernels).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Coeff {
    /// Matrix: 0 = A, 1 = B.
    pub m: u32,
    /// Output row (domain index).
    pub c: u32,
    /// Witness signal.
    pub s: u32,
    pub _pad: u32,
    /// Fr, Montgomery form (4 little-endian u64 limbs).
    pub value: [u64; 4],
}

const _: () = assert!(size_of::<Coeff>() == 48);

// SAFETY: repr(C), u32/u64 fields with an explicit pad word: no padding bytes, any bits valid.
unsafe impl Pod for Coeff {}

const HEADER_BYTES: u64 = 32;

/// The preprocessed coefficients in shared buffers: the CUDA `preprocessed_coeffs`
/// (`d_coeffs` = A then B, `d_independent_indices` = A's then B's row starts).
pub struct Coeffs {
    /// `count_a` coefficients of A, then `count_b` of B.
    pub coeffs: SharedBuf<Coeff>,
    /// `ind_a` row starts of A (relative to A's first coefficient), then `ind_b` of B (relative to
    /// B's first coefficient). Each list ends with its coefficient count, so A has `ind_a - 1`
    /// rows.
    pub indices: SharedBuf<u32>,
    pub count_a: usize,
    pub count_b: usize,
    pub ind_a: usize,
    pub ind_b: usize,
    pub stats: LoadStats,
}

impl Coeffs {
    /// Load `preprocessed_coeffs.bin` (`SetupParams::pcoeffs_path`). Checks the header against
    /// the file size (exactly, as CUDA's compact loader does); [Coeffs::validate] checks the
    /// contents. Takes no lock and needs no `MetalHal`.
    pub fn load(device: &DeviceRef, path: &Path) -> Result<Self> {
        let start = Instant::now();
        let allocated_before = device.current_allocated_size();
        let p = path.display();
        let file = File::open(path).with_context(|| format!("open(\"{p}\")"))?;
        let size = file
            .metadata()
            .with_context(|| format!("fstat(\"{p}\")"))?
            .len();
        ensure!(size >= HEADER_BYTES, "\"{p}\" is truncated");
        let mut hdr = [0u8; 32];
        std::os::unix::fs::FileExt::read_exact_at(&file, &mut hdr, 0)
            .with_context(|| format!("reading \"{p}\""))?;
        let [count_a, count_b, ind_a, ind_b] = limbs_le(&hdr);
        let expect = (count_a as u128 + count_b as u128) * size_of::<Coeff>() as u128
            + (ind_a as u128 + ind_b as u128) * 4
            + HEADER_BYTES as u128;
        ensure!(
            expect == size as u128,
            "\"{p}\": header (coefficients {count_a} + {count_b}, indices {ind_a} + {ind_b}) needs \
             {expect} bytes, the file has {size}"
        );
        ensure!(
            ind_a >= 1 && ind_b >= 1,
            "\"{p}\": each index list needs at least its terminating count"
        );
        let (count_a, count_b, ind_a, ind_b) = (
            count_a as usize,
            count_b as usize,
            ind_a as usize,
            ind_b as usize,
        );

        let mut coeffs = SharedBuf::<Coeff>::alloc(device, "coefficients", count_a + count_b)?;
        read_exact_at_par(&file, HEADER_BYTES, coeffs.as_bytes_mut())
            .with_context(|| format!("reading the coefficients of \"{p}\""))?;
        let mut indices = SharedBuf::<u32>::alloc(device, "coefficient indices", ind_a + ind_b)?;
        read_exact_at_par(
            &file,
            HEADER_BYTES + coeffs.byte_len(),
            indices.as_bytes_mut(),
        )
        .with_context(|| format!("reading the coefficient indices of \"{p}\""))?;

        let stats = LoadStats {
            elapsed: start.elapsed(),
            bytes: coeffs.byte_len() + indices.byte_len(),
            allocated_before,
            allocated_after: device.current_allocated_size(),
        };
        Ok(Self {
            coeffs,
            indices,
            count_a,
            count_b,
            ind_a,
            ind_b,
            stats,
        })
    }

    pub fn coeffs_a(&self) -> &[Coeff] {
        &self.coeffs.as_slice()[..self.count_a]
    }

    pub fn coeffs_b(&self) -> &[Coeff] {
        &self.coeffs.as_slice()[self.count_a..]
    }

    pub fn indices_a(&self) -> &[u32] {
        &self.indices.as_slice()[..self.ind_a]
    }

    pub fn indices_b(&self) -> &[u32] {
        &self.indices.as_slice()[self.ind_a..]
    }

    /// Element offsets for binding B's halves (`as_arg_with_offset`-style): B's coefficients
    /// start at `count_a` coefficients, B's indices at `ind_a` indices.
    pub fn b_offsets(&self) -> (usize, usize) {
        (self.count_a, self.ind_a)
    }

    /// Payload bytes held in GPU buffers.
    pub fn resident_bytes(&self) -> u64 {
        self.coeffs.byte_len() + self.indices.byte_len()
    }

    /// Check what `witness_into_poly` relies on, so a corrupt file cannot make a kernel read or
    /// write out of bounds: A's coefficients have m = 0 and B's m = 1; each index list starts at
    /// 0, increases strictly and ends at its coefficient count; the coefficients of a row share
    /// `c`, and `c` increases strictly from row to row and is below `domain_size`; every `s` is
    /// below `witness_len`; every value is a canonical Fr. Multi-threaded (one pass over 1.4 GB on
    /// the real file).
    pub fn validate(&self, witness_len: usize, domain_size: usize) -> Result<()> {
        validate_part(
            "A",
            0,
            self.coeffs_a(),
            self.indices_a(),
            witness_len,
            domain_size,
        )?;
        validate_part(
            "B",
            1,
            self.coeffs_b(),
            self.indices_b(),
            witness_len,
            domain_size,
        )
    }
}

fn validate_part(
    name: &str,
    m: u32,
    coeffs: &[Coeff],
    ind: &[u32],
    witness_len: usize,
    domain_size: usize,
) -> Result<()> {
    ensure!(
        ind.first() == Some(&0) && ind.last().map(|&l| l as usize) == Some(coeffs.len()),
        "coefficients {name}: indices must run from 0 to the coefficient count {}",
        coeffs.len()
    );
    let rows = ind.len() - 1;
    // Row bounds first (sequential, cheap), so the parallel pass can index freely.
    if let Some(row) = (0..rows).find(|&r| ind[r] >= ind[r + 1]) {
        bail!(
            "coefficients {name}: row {row} is empty or indices decrease ({}..{})",
            ind[row],
            ind[row + 1]
        );
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 16);
    let per = rows.div_ceil(threads).max(1);
    let r_mod = Fr::MODULUS;
    let check_rows = |lo: usize, hi: usize| -> Result<()> {
        for row in lo..hi {
            let (start, end) = (ind[row] as usize, ind[row + 1] as usize);
            let c = coeffs[start].c;
            ensure!(
                (c as usize) < domain_size,
                "coefficients {name}: row {row} writes index {c} >= domain size {domain_size}"
            );
            if row > 0 {
                let prev = coeffs[ind[row - 1] as usize].c;
                ensure!(
                    c > prev,
                    "coefficients {name}: row {row} output {c} does not follow {prev}"
                );
            }
            for (j, co) in coeffs[start..end].iter().enumerate() {
                if co.m != m || co.c != c || co.s as usize >= witness_len {
                    bail!(
                        "coefficients {name}: coefficient {} (row {row}) is {co:?}; expected m = \
                         {m}, c = {c}, s < {witness_len}",
                        start + j
                    );
                }
                ensure!(
                    BigInt::new(co.value) < r_mod,
                    "coefficients {name}: coefficient {} value is not a canonical scalar",
                    start + j
                );
            }
        }
        Ok(())
    };
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..rows)
            .step_by(per)
            .map(|lo| {
                let check_rows = &check_rows;
                s.spawn(move || check_rows(lo, (lo + per).min(rows)))
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|h| h.join().expect("coefficient validation thread panicked"))
    })
}

// ---------------------------------------------------------------------------------------------
// Fuzzed MSM results
// ---------------------------------------------------------------------------------------------

/// The fuzzed results file: sppark's `msm_results {point_t a, b_g1, c, h; point_fp2_t b_g2}`
/// (Jacobian, Montgomery), already negated: the MSMs of the SRS points with the deterministic
/// ChaCha fuzz (the CUDA `generate_random_scalars`), which the prover adds back to remove the fuzz
/// it added to the witness. `h` is junk: CUDA never fuzzes H and the setup wrote an
/// uninitialised point (in the shipped file it is not on the curve); never use it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FuzzedResults {
    pub a: G1JacobianRaw,
    pub b_g1: G1JacobianRaw,
    pub c: G1JacobianRaw,
    pub h: G1JacobianRaw,
    pub b_g2: G2JacobianRaw,
}

pub const FUZZED_RESULTS_BYTES: usize = 576;
const _: () = assert!(size_of::<FuzzedResults>() == FUZZED_RESULTS_BYTES);

/// [FuzzedResults] as arkworks points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuzzedResultsArk {
    pub a: G1Projective,
    pub b_g1: G1Projective,
    pub c: G1Projective,
    pub h: G1Projective,
    pub b_g2: G2Projective,
}

impl FuzzedResults {
    /// Read `fuzzed_msm_results.bin` (`SetupParams::fres_path`). The file must be exactly 576
    /// bytes (CUDA only requires that many).
    pub fn load(path: &Path) -> Result<Self> {
        let p = path.display();
        let bytes = std::fs::read(path).with_context(|| format!("open(\"{p}\")"))?;
        ensure!(
            bytes.len() == FUZZED_RESULTS_BYTES,
            "fuzzed results file \"{p}\" is not the correct size: expected \
             {FUZZED_RESULTS_BYTES}, got {}",
            bytes.len()
        );
        Ok(Self::from_bytes(bytes.as_slice().try_into().unwrap()))
    }

    pub fn from_bytes(b: &[u8; FUZZED_RESULTS_BYTES]) -> Self {
        let fq = |i: usize| limbs_le(&b[32 * i..32 * i + 32]);
        let g1 = |k: usize| G1JacobianRaw {
            x: fq(3 * k),
            y: fq(3 * k + 1),
            z: fq(3 * k + 2),
        };
        let fq2 = |i: usize| [fq(i), fq(i + 1)];
        Self {
            a: g1(0),
            b_g1: g1(1),
            c: g1(2),
            h: g1(3),
            b_g2: G2JacobianRaw {
                x: fq2(12),
                y: fq2(14),
                z: fq2(16),
            },
        }
    }

    pub fn to_ark(&self) -> Result<FuzzedResultsArk> {
        Ok(FuzzedResultsArk {
            a: self.a.to_ark()?,
            b_g1: self.b_g1.to_ark()?,
            c: self.c.to_ark()?,
            h: self.h.to_ark()?,
            b_g2: self.b_g2.to_ark()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{os::unix::fs::FileExt, time::Instant};

    use ark_ec::{CurveGroup, PrimeGroup};
    use ark_ff::UniformRand;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::{super::srs, *};
    use crate::metal::srs::tests::{component_dir, device, temp_path};

    fn coeff_bytes(c: &Coeff) -> Vec<u8> {
        let mut v = Vec::with_capacity(48);
        for w in [c.m, c.c, c.s, c._pad] {
            v.extend(w.to_le_bytes());
        }
        for w in c.value {
            v.extend(w.to_le_bytes());
        }
        v
    }

    /// A small preprocessed file: rows of A then B, in the CUDA constructor's layout.
    fn synthetic(
        rows_a: &[(u32, usize)],
        rows_b: &[(u32, usize)],
        witness: u32,
    ) -> (Vec<u8>, Vec<Coeff>, Vec<u32>) {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let mut coeffs = Vec::new();
        let mut indices = Vec::new();
        for (m, rows) in [(0u32, rows_a), (1, rows_b)] {
            let base = coeffs.len();
            for &(c, n) in rows {
                indices.push((coeffs.len() - base) as u32);
                for _ in 0..n {
                    let s = (rand::Rng::gen::<u32>(&mut rng)) % witness;
                    coeffs.push(Coeff {
                        m,
                        c,
                        s,
                        _pad: 0,
                        value: Fr::rand(&mut rng).0 .0,
                    });
                }
            }
            indices.push((coeffs.len() - base) as u32);
        }
        let count_a: usize = rows_a.iter().map(|r| r.1).sum();
        let mut bytes = Vec::new();
        for h in [
            count_a,
            coeffs.len() - count_a,
            rows_a.len() + 1,
            rows_b.len() + 1,
        ] {
            bytes.extend((h as u64).to_le_bytes());
        }
        bytes.extend(coeffs.iter().flat_map(coeff_bytes));
        bytes.extend(indices.iter().flat_map(|i| i.to_le_bytes()));
        (bytes, coeffs, indices)
    }

    fn load_bytes(bytes: &[u8]) -> Result<Coeffs> {
        let path = temp_path("coeffs.bin");
        std::fs::write(&path, bytes).unwrap();
        let res = Coeffs::load(&device(), &path);
        std::fs::remove_file(&path).unwrap();
        res
    }

    #[test]
    fn synthetic_coeffs_load_and_validate() {
        let (bytes, coeffs, indices) = synthetic(&[(0, 3), (2, 1), (7, 5)], &[(1, 2), (2, 4)], 50);
        let c = load_bytes(&bytes).unwrap();
        assert_eq!((c.count_a, c.count_b, c.ind_a, c.ind_b), (9, 6, 4, 3));
        assert_eq!(c.coeffs.as_slice(), coeffs.as_slice());
        assert_eq!(c.indices.as_slice(), indices.as_slice());
        assert_eq!(c.indices_a(), &[0, 3, 4, 9]);
        assert_eq!(c.indices_b(), &[0, 2, 6]);
        assert_eq!(c.coeffs_b()[0].m, 1);
        assert_eq!(c.b_offsets(), (9, 4));
        assert_eq!(c.resident_bytes(), 15 * 48 + 7 * 4);
        c.validate(50, 8).unwrap();
        // Bounds are enforced.
        assert!(format!("{:#}", c.validate(50, 7).unwrap_err()).contains("domain size"));
        let max_s = coeffs.iter().map(|c| c.s).max().unwrap() as usize;
        assert!(c.validate(max_s, 8).is_err());
        c.validate(max_s + 1, 8).unwrap();
    }

    #[test]
    fn malformed_coeffs_are_rejected() {
        let (good, _, _) = synthetic(&[(0, 3), (2, 1), (7, 5)], &[(1, 2), (2, 4)], 50);
        let coeff_at = |i: usize| 32 + 48 * i;
        let ind_at = |i: usize| 32 + 48 * 15 + 4 * i;
        // Load-time (header) failures.
        let mut b = good.clone();
        b.pop();
        assert!(format!("{:#}", load_bytes(&b).err().unwrap()).contains("needs"));
        let mut b = good.clone();
        b[0] = 10; // count_a 9 -> 10
        assert!(load_bytes(&b).is_err());
        assert!(load_bytes(&good[..16]).is_err());
        // Content failures found by validate().
        let cases: Vec<(Box<dyn Fn(&mut Vec<u8>)>, &str)> = vec![
            (Box::new(move |b| b[coeff_at(0)] = 1), "expected m = 0"),
            (Box::new(move |b| b[coeff_at(10)] = 0), "expected m = 1"),
            (
                Box::new(move |b| b[coeff_at(1) + 4] = 1),
                "expected m = 0, c = 0",
            ),
            (Box::new(move |b| b[coeff_at(3) + 4] = 0), "does not follow"),
            (
                Box::new(move |b| {
                    b[coeff_at(2) + 8..coeff_at(2) + 12].copy_from_slice(&50u32.to_le_bytes())
                }),
                "s < 50",
            ),
            (
                Box::new(move |b| b[coeff_at(4) + 16..coeff_at(4) + 48].fill(0xff)),
                "canonical",
            ),
            (Box::new(move |b| b[ind_at(0)] = 1), "from 0"),
            (
                Box::new(move |b| b[ind_at(1)] = 4),
                "empty or indices decrease",
            ),
            (
                Box::new(move |b| b[ind_at(6)] = 5),
                "from 0 to the coefficient count",
            ),
        ];
        for (i, (mutate, want)) in cases.iter().enumerate() {
            let mut b = good.clone();
            mutate(&mut b);
            let c = load_bytes(&b).unwrap();
            let err = format!(
                "{:#}",
                c.validate(50, 8)
                    .err()
                    .unwrap_or_else(|| panic!("case {i} validated"))
            );
            assert!(err.contains(want), "case {i}: {err:?} lacks {want:?}");
        }
    }

    #[test]
    fn fuzzed_results_layout() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let pts1: Vec<G1Projective> = (0..4)
            .map(|_| G1Projective::generator() * Fr::rand(&mut rng))
            .collect();
        let p2 = G2Projective::generator() * Fr::rand(&mut rng);
        let mut bytes = Vec::new();
        for p in &pts1 {
            for f in [p.x, p.y, p.z] {
                bytes.extend(f.0 .0.iter().flat_map(|w| w.to_le_bytes()));
            }
        }
        for f in [p2.x, p2.y, p2.z] {
            for c in [f.c0, f.c1] {
                bytes.extend(c.0 .0.iter().flat_map(|w| w.to_le_bytes()));
            }
        }
        let path = temp_path("fres.bin");
        std::fs::write(&path, &bytes).unwrap();
        let fr = FuzzedResults::load(&path).unwrap().to_ark().unwrap();
        assert_eq!([fr.a, fr.b_g1, fr.c, fr.h], pts1.as_slice());
        assert_eq!(fr.b_g2, p2);
        std::fs::write(&path, &bytes[..575]).unwrap();
        assert!(FuzzedResults::load(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }

    /// The real files: the coefficients validate against the zkey's sizes, A and B are exactly
    /// zkey section 4's records (in file order, split by matrix), and the fuzzed results are
    /// valid curve points.
    #[test]
    fn component_coeffs_and_fuzz() {
        let Some(dir) = component_dir() else { return };
        let header = srs::read_zkey_header(&dir.join("stark_verify_final.zkey")).unwrap();
        let dev = device();
        let rss0 = srs::tests::rss_bytes();
        let c = Coeffs::load(&dev, &dir.join("preprocessed_coeffs.bin")).unwrap();
        println!(
            "process RSS {:.3} -> {:.3} GB",
            rss0 as f64 / 1e9,
            srs::tests::rss_bytes() as f64 / 1e9
        );
        let st = c.stats;
        let t = Instant::now();
        c.validate(header.witness_len(), header.domain_size)
            .unwrap();
        println!(
            "coeffs: {} A + {} B, {} + {} indices; {:.3} GB loaded in {:.3} s ({:.2} GB/s), \
             validated in {:.3} s; device allocated {:.3} -> {:.3} GB",
            c.count_a,
            c.count_b,
            c.ind_a,
            c.ind_b,
            c.resident_bytes() as f64 / 1e9,
            st.elapsed.as_secs_f64(),
            st.gb_per_s(),
            t.elapsed().as_secs_f64(),
            st.allocated_before as f64 / 1e9,
            st.allocated_after as f64 / 1e9,
        );
        assert_eq!(c.count_a + c.count_b, header.num_coeffs);

        // Cross-check against the zkey's own coefficient section (44-byte packed records).
        let sec = header.sections[srs::section::COEFFS];
        let mut raw = vec![0u8; sec.size as usize - 4];
        let zkey = File::open(dir.join("stark_verify_final.zkey")).unwrap();
        zkey.read_exact_at(&mut raw, sec.offset + 4).unwrap();
        let (mut ia, mut ib) = (0, 0);
        for rec in raw.chunks_exact(44) {
            let u = |o: usize| u32::from_le_bytes(rec[o..o + 4].try_into().unwrap());
            let value = limbs_le(&rec[12..44]);
            let got = if u(0) == 0 {
                ia += 1;
                c.coeffs_a()[ia - 1]
            } else {
                ib += 1;
                c.coeffs_b()[ib - 1]
            };
            assert_eq!((got.m, got.c, got.s, got.value), (u(0), u(4), u(8), value));
        }
        assert_eq!((ia, ib), (c.count_a, c.count_b));

        let fres = FuzzedResults::load(&dir.join("fuzzed_msm_results.bin")).unwrap();
        let fa = fres.to_ark().unwrap();
        for p in [fa.a, fa.b_g1, fa.c] {
            let p = p.into_affine();
            assert!(!p.infinity && p.is_on_curve() && p.is_in_correct_subgroup_assuming_on_curve());
        }
        let b2 = fa.b_g2.into_affine();
        assert!(!b2.infinity && b2.is_on_curve() && b2.is_in_correct_subgroup_assuming_on_curve());
        // CUDA never fuzzes H: the setup wrote an uninitialised `point_t` there. In the shipped
        // file it is not a curve point, so it must never be used (or converted with `to_ark`
        // and trusted).
        let h = fa.h.into_affine();
        println!("fuzzed h = {:?} (on curve: {})", fres.h, h.is_on_curve());
        assert!(!h.is_on_curve());
    }
}
