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

//! The Metal prover's zkey parser (risc0-groth16-sys `metal::srs`, item G3) against the
//! verifying key risc0 verifies with: the key in the rzup component's `stark_verify_final.zkey`,
//! decoded by reinterpreting its Montgomery limbs as arkworks field elements, must equal
//! `risc0_groth16::verifying_key()` (which settles the ark/sppark Montgomery compatibility).
//! Lives here because risc0-groth16-sys cannot depend on risc0-groth16.

#![cfg(all(feature = "metal", target_os = "macos"))]

use std::path::PathBuf;

use ark_bn254::Bn254;
use ark_serialize::CanonicalDeserialize;
use risc0_groth16_sys::metal::srs;

fn component_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("RISC0_GROTH16_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join(".risc0/extensions/v0.1.0-risc0-groth16"))
        })?;
    if dir.join("stark_verify_final.zkey").is_file() {
        Some(dir)
    } else {
        eprintln!(
            "skipped: no risc0-groth16 component at {} (rzup install risc0-groth16)",
            dir.display()
        );
        None
    }
}

/// `risc0_groth16::VerifyingKey`'s inner arkworks key (its serde form is the uncompressed
/// arkworks serialization).
fn risc0_vk() -> ark_groth16::VerifyingKey<Bn254> {
    let json = serde_json::to_value(risc0_groth16::verifying_key()).unwrap();
    let bytes: Vec<u8> = serde_json::from_value(json).unwrap();
    ark_groth16::VerifyingKey::<Bn254>::deserialize_uncompressed(bytes.as_slice()).unwrap()
}

#[test]
fn zkey_verifying_key_matches_risc0() {
    let Some(dir) = component_dir() else { return };
    let header = srs::read_zkey_header(&dir.join("stark_verify_final.zkey")).unwrap();
    let parsed = header.vk.to_ark().unwrap();
    let want = risc0_vk();

    assert_eq!(parsed.alpha_g1, want.alpha_g1, "alpha_g1");
    assert_eq!(parsed.beta_g2, want.beta_g2, "beta_g2");
    assert_eq!(parsed.gamma_g2, want.gamma_g2, "gamma_g2");
    assert_eq!(parsed.delta_g2, want.delta_g2, "delta_g2");
    assert_eq!(parsed.ic, want.gamma_abc_g1, "IC / gamma_abc_g1");
    assert_eq!(header.num_public + 1, want.gamma_abc_g1.len());

    // And back: the GPU-layout bytes of risc0's key are the zkey's bytes.
    assert_eq!(
        header.vk.alpha_g1,
        srs::G1AffineRaw::from_ark(&want.alpha_g1)
    );
    assert_eq!(
        header.vk.delta_g2,
        srs::G2AffineRaw::from_ark(&want.delta_g2)
    );
}
