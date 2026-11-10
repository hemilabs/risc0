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

//! Layout of the preflight structs shared with the Metal kernels (plan 4.2, item R1).
//!
//! `kernels/metal/preflight.h` static_asserts the MSL sizes and offsets of `PreflightCycle` and
//! `MemoryTransaction`; this test checks the Rust side (`RawPreflightCycle`,
//! `RawMemoryTransaction`) against the same numbers, read from that header, so the two sides
//! cannot drift apart silently. Runs on every target (no GPU or toolchain needed).

use std::{collections::BTreeMap, mem::offset_of};

use risc0_circuit_rv32im_sys::{RawMemoryTransaction, RawPreflightCycle};

const PREFLIGHT_H: &str = include_str!("../kernels/metal/preflight.h");

/// (sizes, offsets) asserted in preflight.h: `RV32IM_ASSERT_SIZE(T, N);` and
/// `RV32IM_ASSERT_OFFSET(T, field, N);`, one per line.
fn msl_asserts() -> (BTreeMap<String, usize>, BTreeMap<(String, String), usize>) {
    let mut sizes = BTreeMap::new();
    let mut offsets = BTreeMap::new();
    for line in PREFLIGHT_H.lines().map(str::trim) {
        let args = |prefix: &str| -> Option<Vec<String>> {
            let rest = line.strip_prefix(prefix)?.strip_suffix(");")?;
            Some(rest.split(',').map(|s| s.trim().to_string()).collect())
        };
        if let Some(a) = args("RV32IM_ASSERT_SIZE(") {
            assert_eq!(a.len(), 2, "{line}");
            sizes.insert(a[0].clone(), a[1].parse().unwrap());
        } else if let Some(a) = args("RV32IM_ASSERT_OFFSET(") {
            assert_eq!(a.len(), 3, "{line}");
            offsets.insert((a[0].clone(), a[1].clone()), a[2].parse().unwrap());
        }
    }
    (sizes, offsets)
}

#[test]
fn raw_preflight_sizes() {
    // The numbers the Metal port relies on (PLAN.md 4.2: 36 and 20 bytes, verified).
    assert_eq!(size_of::<RawPreflightCycle>(), 36);
    assert_eq!(size_of::<RawMemoryTransaction>(), 20);
    assert_eq!(align_of::<RawPreflightCycle>(), 4);
    assert_eq!(align_of::<RawMemoryTransaction>(), 4);
    // Arrays are memcpy'd as-is, so the stride must equal the size.
    assert_eq!(size_of::<[RawPreflightCycle; 3]>(), 3 * 36);
    assert_eq!(size_of::<[RawMemoryTransaction; 3]>(), 3 * 20);
}

#[test]
fn raw_preflight_matches_msl() {
    let (sizes, offsets) = msl_asserts();
    assert_eq!(
        sizes.get("PreflightCycle"),
        Some(&size_of::<RawPreflightCycle>())
    );
    assert_eq!(
        sizes.get("MemoryTransaction"),
        Some(&size_of::<RawMemoryTransaction>())
    );

    // MSL field name -> Rust offset.
    let cycle: &[(&str, usize)] = &[
        ("state", offset_of!(RawPreflightCycle, state)),
        ("pc", offset_of!(RawPreflightCycle, pc)),
        ("major", offset_of!(RawPreflightCycle, major)),
        ("minor", offset_of!(RawPreflightCycle, minor)),
        ("machineMode", offset_of!(RawPreflightCycle, machine_mode)),
        ("padding", offset_of!(RawPreflightCycle, padding)),
        ("userCycle", offset_of!(RawPreflightCycle, user_cycle)),
        ("txnIdx", offset_of!(RawPreflightCycle, txn_idx)),
        ("pagingIdx", offset_of!(RawPreflightCycle, paging_idx)),
        ("bigintIdx", offset_of!(RawPreflightCycle, bigint_idx)),
        ("diffCount", offset_of!(RawPreflightCycle, diff_count)),
    ];
    let txn: &[(&str, usize)] = &[
        ("addr", offset_of!(RawMemoryTransaction, addr)),
        ("cycle", offset_of!(RawMemoryTransaction, cycle)),
        ("word", offset_of!(RawMemoryTransaction, word)),
        ("prevCycle", offset_of!(RawMemoryTransaction, prev_cycle)),
        ("prevWord", offset_of!(RawMemoryTransaction, prev_word)),
    ];
    let mut expected = BTreeMap::new();
    for (ty, fields) in [("PreflightCycle", cycle), ("MemoryTransaction", txn)] {
        for (field, off) in fields {
            expected.insert((ty.to_string(), field.to_string()), *off);
        }
    }
    // Every field asserted on both sides, with the same offset.
    assert_eq!(offsets, expected);
}
