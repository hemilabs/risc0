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

//! Item C16 (c) (docs/metal-port/notes/C16.md): in auto mode, `prove` with a Groth16 receipt
//! releases the segment prover before recursion.
//!
//! `c16_groth16_release_ab` proves a BusyLoop guest to a Groth16 receipt `C16_REPS` times (default
//! 8) in one process, alternating `RISC0_METAL_RELEASE_SEG_BEFORE_REC` between `budget` (A, the
//! pre-C16 auto mode) and unset (B, the default), starting with A, and prints `[C16AB]` lines
//! (wall time per proof). The `[succinct_to_groth16]` lines printed by each proof carry the
//! identity_p254 / shrink_wrap split. Knobs: `C16_PO2` (20), `C16_CYCLES` (2^22), `C16_QUIET=1`
//! (wait for a quiet GPU before each proof). Every Groth16 receipt is verified.

use std::time::Instant;

use risc0_zkvm_methods::{multi_test::MultiTestSpec, MULTI_TEST_ELF, MULTI_TEST_ID};

use super::prover_impl::{RELEASE_SEG_BEFORE_REC_ENV, SEG_RELEASES};
use crate::{get_prover_server, ExecutorEnv, InnerReceipt, ProverOpts};

/// The GPU's "Device Utilization %" (IORegistry), or None.
fn gpu_utilization() -> Option<u32> {
    let out = std::process::Command::new("ioreg")
        .args(["-r", "-d", "1", "-c", "IOAccelerator"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let key = "\"Device Utilization %\"=";
    let i = text.find(key)? + key.len();
    text[i..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .ok()
}

/// With `C16_QUIET=1`, wait (up to 60 s) for two readings 250 ms apart below 50 % GPU busy.
fn wait_gpu_quiet() {
    if !std::env::var("C16_QUIET").is_ok_and(|v| v == "1") {
        return;
    }
    let t0 = Instant::now();
    let mut quiet = 0;
    while t0.elapsed().as_secs() < 60 && quiet < 2 {
        match gpu_utilization() {
            Some(u) if u >= 50 => quiet = 0,
            _ => quiet += 1,
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "sets process environment and times whole Groth16 proofs; run alone"]
fn c16_groth16_release_ab() {
    let po2 = env_u64("C16_PO2", 20) as u32;
    let cycles = env_u64("C16_CYCLES", 1 << 22);
    let reps = env_u64("C16_REPS", 8);
    for rep in 0..reps {
        let arm = if rep % 2 == 0 {
            std::env::set_var(RELEASE_SEG_BEFORE_REC_ENV, "budget");
            "A"
        } else {
            std::env::remove_var(RELEASE_SEG_BEFORE_REC_ENV);
            "B"
        };
        let env = ExecutorEnv::builder()
            .write(&MultiTestSpec::BusyLoop { cycles })
            .unwrap()
            .segment_limit_po2(po2)
            .build()
            .unwrap();
        let releases = SEG_RELEASES.with(|n| n.get());
        wait_gpu_quiet();
        let t = Instant::now();
        let receipt = get_prover_server(&ProverOpts::groth16())
            .unwrap()
            .prove(env, MULTI_TEST_ELF)
            .unwrap()
            .receipt;
        let s = t.elapsed().as_secs_f64();
        let released = SEG_RELEASES.with(|n| n.get()) - releases;
        assert!(matches!(receipt.inner, InnerReceipt::Groth16(_)));
        receipt.verify(MULTI_TEST_ID).unwrap();
        eprintln!("[C16AB] rep={rep} arm={arm} prove={s:.3}s seg-release={released} verified");
        if arm == "B" {
            assert_eq!(released, 1, "auto mode did not release before recursion");
        }
    }
    std::env::remove_var(RELEASE_SEG_BEFORE_REC_ENV);
}
