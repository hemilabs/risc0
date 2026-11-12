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

//! Capture of real recursion proofs (lift, join, resolve, identity_p254) for the C11 profile and
//! A/B harnesses (measurement aid, off by default).
//!
//! `RISC0_METAL_REC_CAPTURE=<dir>`: every Metal recursion proof first writes its program, input
//! and (when the caller prepared one) preflight trace to `<dir>/NNNN_<hashfn>_po2P_rowsR.rcap`,
//! so `c11_tests` can replay the exact lift/join/resolve shapes of a zkvm run inside this crate.
//! The file is raw host words (same machine only). Never set it in production: it writes every
//! recursion input to disk.

// The reader side is used by the C11 test harness only.
#![cfg_attr(not(test), allow(dead_code))]

use std::{
    collections::VecDeque,
    io::Write as _,
    path::{Path, PathBuf},
    sync::{atomic::AtomicUsize, atomic::Ordering, Arc},
};

use anyhow::{bail, ensure, Context as _, Result};
use risc0_circuit_recursion_sys::RawPreflightCycle;
use risc0_zkp::field::baby_bear::{BabyBearElem, BabyBearExtElem};

use crate::prove::{Preflight, Program, RecursionProver, RecursionReceipt};

/// The knob: a directory to write captures into.
pub(crate) const CAPTURE_ENV: &str = "RISC0_METAL_REC_CAPTURE";

const MAGIC: u32 = 0x5243_4331; // "RCC1"

static SEQ: AtomicUsize = AtomicUsize::new(0);

/// A captured proof.
pub(crate) struct Capture {
    pub hashfn: String,
    pub program: Arc<Program>,
    pub input: VecDeque<u32>,
    /// `(wom, cycles, iops, output)` when the caller passed a prepared preflight.
    pub trace: Option<(
        Vec<BabyBearExtElem>,
        Vec<RawPreflightCycle>,
        Vec<BabyBearExtElem>,
        Vec<u32>,
    )>,
}

impl Capture {
    /// A fresh preflight for a replay (none if the capture had none: the prover runs it).
    pub fn preflight(&self) -> Option<Preflight> {
        self.trace.as_ref().map(|(wom, cycles, iops, output)| {
            let mut p = Preflight::new(VecDeque::new());
            p.trace.wom = wom.clone();
            p.trace.cycles = cycles
                .iter()
                .map(|c| RawPreflightCycle {
                    iop_idx: c.iop_idx,
                    is_par_safe: c.is_par_safe,
                })
                .collect();
            p.trace.iops = iops.clone();
            p.output = output.clone();
            p
        })
    }
}

fn words<T>(v: &[T]) -> &[u32] {
    // SAFETY: only used for the u32-word types below (BabyBearElem, BabyBearExtElem,
    // RawPreflightCycle are repr(transparent)/repr(C) over u32 words, no padding).
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u32, std::mem::size_of_val(v) / 4) }
}

fn from_words<T: Clone>(w: &[u32]) -> Vec<T> {
    let n = std::mem::size_of_val(w) / std::mem::size_of::<T>();
    // SAFETY: as in [words]; any bit pattern was a valid T when it was written.
    unsafe { std::slice::from_raw_parts(w.as_ptr() as *const T, n) }.to_vec()
}

fn hash_id(h: &str) -> u32 {
    match h {
        "poseidon2" => 0,
        "sha-256" => 1,
        _ => 2,
    }
}

fn hash_name(id: u32) -> &'static str {
    match id {
        0 => "poseidon2",
        1 => "sha-256",
        _ => "poseidon_254",
    }
}

fn write(
    dir: &Path,
    hashfn: &str,
    program: &Program,
    input: &VecDeque<u32>,
    preflight: Option<&Preflight>,
) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!(
        "{n:04}_{hashfn}_po2{}_rows{}.rcap",
        program.po2,
        program.code_rows()
    ));
    let input: Vec<u32> = input.iter().copied().collect();
    let code: Vec<u32> = program.code.iter().map(|e| e.as_u32_montgomery()).collect();
    let (wom, cycles, iops, output): (&[u32], &[u32], &[u32], &[u32]) = match preflight {
        Some(p) => (
            words(&p.trace.wom),
            words(&p.trace.cycles),
            words(&p.trace.iops),
            &p.output,
        ),
        None => (&[], &[], &[], &[]),
    };
    let header = [
        MAGIC,
        hash_id(hashfn),
        program.po2 as u32,
        program.code_size as u32,
        preflight.is_some() as u32,
        code.len() as u32,
        input.len() as u32,
        wom.len() as u32,
        cycles.len() as u32,
        iops.len() as u32,
        output.len() as u32,
    ];
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
    for part in [&header[..], &code, &input, wom, cycles, iops, output] {
        f.write_all(bytemuck::cast_slice(part))?;
    }
    f.flush()?;
    Ok(path)
}

/// Read a capture written by [CaptureProver].
pub(crate) fn read(path: &Path) -> Result<Capture> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(bytes.len() % 4 == 0 && bytes.len() >= 44, "short capture");
    let w: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if w[0] != MAGIC {
        bail!("{}: not a capture", path.display());
    }
    let lens = &w[5..11];
    let mut at = 11usize;
    let mut take = |n: u32| {
        let s = &w[at..at + n as usize];
        at += n as usize;
        s
    };
    let code = take(lens[0]);
    let input = take(lens[1]);
    let (wom, cycles, iops, output) = (take(lens[2]), take(lens[3]), take(lens[4]), take(lens[5]));
    let program = Program {
        code: code.iter().map(|&x| BabyBearElem::new_raw(x)).collect(),
        code_size: w[3] as usize,
        po2: w[2] as usize,
    };
    Ok(Capture {
        hashfn: hash_name(w[1]).to_string(),
        program: Arc::new(program),
        input: input.iter().copied().collect(),
        trace: (w[4] == 1).then(|| {
            (
                from_words(wom),
                from_words(cycles),
                from_words(iops),
                output.to_vec(),
            )
        }),
    })
}

/// The Metal prover under [CAPTURE_ENV].
pub(crate) struct CaptureProver {
    pub inner: Box<dyn RecursionProver>,
    pub dir: PathBuf,
    pub hashfn: String,
}

impl CaptureProver {
    fn capture(&self, program: &Program, input: &VecDeque<u32>, preflight: Option<&Preflight>) {
        match write(&self.dir, &self.hashfn, program, input, preflight) {
            Ok(p) => eprintln!("[{CAPTURE_ENV}] wrote {}", p.display()),
            Err(e) => tracing::warn!("{CAPTURE_ENV}: {e:#}"),
        }
    }
}

impl RecursionProver for CaptureProver {
    fn prove(
        &self,
        program: &Program,
        input: VecDeque<u32>,
        preflight: Option<Preflight>,
    ) -> Result<RecursionReceipt> {
        self.capture(program, &input, preflight.as_ref());
        self.inner.prove(program, input, preflight)
    }

    fn prewarm(&self, program: &Arc<Program>) -> Result<()> {
        self.inner.prewarm(program)
    }

    fn prove_shared(
        &self,
        program: &Arc<Program>,
        input: VecDeque<u32>,
        preflight: Option<Preflight>,
    ) -> Result<RecursionReceipt> {
        self.capture(program, &input, preflight.as_ref());
        self.inner.prove_shared(program, input, preflight)
    }
}
