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

//! Z7: `KERNEL_NAMES` is exactly the set of kernels in the zkp metallib, and the retired legacy
//! kernels are gone from both.

use std::collections::BTreeSet;

use metal::Device;

use super::{KERNEL_NAMES, METAL_LIB};

/// The pre-port Metal kernels retired in Z7 (plan 4.1, legacy row) plus the dead
/// `eltwise_zeroize_fpext`.
const RETIRED: &[&str] = &[
    "batch_expand",
    "eltwise_zeroize_fpext",
    "multi_bit_reverse",
    "multi_ntt_fwd_step",
    "multi_ntt_rev_step",
    "multi_poly_eval",
    "poseidon2_fold",
    "poseidon2_rows",
    "zk_shift",
];

#[test]
fn kernel_names_match_metallib() {
    let device = Device::system_default().expect("no Metal device found");
    let library = device.new_library_with_data(METAL_LIB).unwrap();
    let in_lib: BTreeSet<String> = library.function_names().into_iter().collect();
    let listed: BTreeSet<String> = KERNEL_NAMES.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        listed.len(),
        KERNEL_NAMES.len(),
        "duplicate entries in KERNEL_NAMES"
    );
    let unlisted: Vec<_> = in_lib.difference(&listed).collect();
    let missing: Vec<_> = listed.difference(&in_lib).collect();
    assert!(
        unlisted.is_empty() && missing.is_empty(),
        "metallib kernels not in KERNEL_NAMES: {unlisted:?}; KERNEL_NAMES not in metallib: {missing:?}"
    );
    for name in RETIRED {
        assert!(!in_lib.contains(*name), "retired kernel {name} still built");
    }
}
