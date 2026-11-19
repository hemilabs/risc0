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

use crate::host::client::env::ProveKeccakRequest;
use anyhow::{bail, Result};
use risc0_binfmt::ByteAddr;
use risc0_circuit_keccak::{max_keccak_inputs, KeccakState, KECCAK_DEFAULT_PO2, KECCAK_PO2_RANGE};
use risc0_circuit_rv32im::execute::MAX_IO_BYTES;
use risc0_zkvm_platform::syscall::{
    keccak_mode::{KECCAK_PERMUTE, KECCAK_PROVE},
    reg_abi::*,
};

use super::{Syscall, SyscallContext, SyscallKind};

/// `RISC0_KECCAK_RIGHTSIZE_PO2=0` proves every keccak request at the session's keccak max po2
/// (upstream 3.1, and the pre-K10 behaviour); set to anything else, at the smallest po2 that
/// holds its inputs ([request_po2]). Unset: [KECCAK_RIGHTSIZE_DEFAULT].
pub(crate) const KECCAK_RIGHTSIZE_ENV: &str = "RISC0_KECCAK_RIGHTSIZE_PO2";

/// W10-GATE: K10 right-sizing is on by default only in Metal builds. Every other build (CPU,
/// CUDA, ROCm, Intel) asks for the session's keccak max po2 by default, exactly as upstream
/// release-3.1, and right-sizes only when [KECCAK_RIGHTSIZE_ENV] opts in.
pub(crate) const KECCAK_RIGHTSIZE_DEFAULT: bool = METAL_BUILD;

/// A Metal build (the `metal` feature, which only builds on macOS/iOS).
const METAL_BUILD: bool = cfg!(all(
    feature = "metal",
    any(target_os = "macos", target_os = "ios")
));

/// The po2 a keccak request of `inputs` permutations is proved at (K10, LEADS #51): the smallest
/// po2 of [KECCAK_PO2_RANGE] whose circuit holds them (`max_keccak_inputs`, the same capacity the
/// batch-full signal to the guest uses), never above `max_po2`. The claim digest does not depend
/// on the po2, and the lift's control ID is one of `KECCAK_CONTROL_IDS`, all under
/// `KECCAK_CONTROL_ROOT`, so a smaller proof verifies exactly as a full-size one; it is up to
/// 2^(max_po2 - 14) times smaller (po2 17: ~0.8 s of prove + lift for a full one, ~0.13 s at 14).
pub(crate) fn request_po2(inputs: usize, max_po2: usize, rightsize: bool) -> usize {
    if !rightsize {
        return max_po2;
    }
    (*KECCAK_PO2_RANGE.start()..max_po2)
        .find(|&po2| max_keccak_inputs(po2) >= inputs)
        .unwrap_or(max_po2)
}

/// Whether keccak requests are right-sized: [KECCAK_RIGHTSIZE_DEFAULT] when
/// [KECCAK_RIGHTSIZE_ENV] is unset, off when it is `0`, on for any other value.
pub(crate) fn rightsize_enabled() -> bool {
    rightsize_from(std::env::var(KECCAK_RIGHTSIZE_ENV).ok().as_deref())
}

fn rightsize_from(knob: Option<&str>) -> bool {
    match knob {
        None => KECCAK_RIGHTSIZE_DEFAULT,
        Some(v) => v.trim() != "0",
    }
}

#[derive(Clone, Default)]
pub(crate) struct SysKeccak {
    inputs: Vec<KeccakState>,
    max_po2: usize,
    max_inputs: usize,
}

impl Syscall for SysKeccak {
    fn syscall(
        &mut self,
        _syscall: &str,
        ctx: &mut dyn SyscallContext,
        to_guest: &mut [u32],
    ) -> Result<(u32, u32)> {
        let mode = ctx.load_register(REG_A3);
        if mode == KECCAK_PERMUTE {
            self.keccak_permute(ctx, to_guest)
        } else if mode == KECCAK_PROVE {
            self.keccak_prove(ctx, to_guest)
        } else {
            bail!("sys_keccak: invalid mode: {mode}")
        }
    }
}

impl SysKeccak {
    pub fn new(max_po2: Option<u32>) -> Self {
        let max_po2 = max_po2.unwrap_or(KECCAK_DEFAULT_PO2 as u32) as usize;

        Self {
            inputs: vec![],
            max_po2,
            max_inputs: max_keccak_inputs(max_po2),
        }
    }

    fn is_full(&self) -> bool {
        self.inputs.len() >= self.max_inputs
    }

    fn keccak_permute(
        &mut self,
        ctx: &mut dyn SyscallContext,
        to_guest: &mut [u32],
    ) -> Result<(u32, u32)> {
        // if we are full at this point, it means that the guest forgot to call prove.
        if self.is_full() {
            bail!("keccak batch is full, prove must be called");
        }

        if to_guest.len() >= MAX_IO_BYTES as usize {
            bail!("invalid sys_keccak_permute");
        }

        let buf_ptr = ByteAddr(ctx.load_register(REG_A4));
        let from_guest = &ctx.load_region(buf_ptr, 25 * 8)?;
        let mut from_guest: KeccakState = bytemuck::cast_slice(from_guest).try_into()?;
        self.inputs.push(from_guest);

        keccak::f1600(&mut from_guest);
        to_guest.clone_from_slice(bytemuck::cast_slice(&from_guest));

        let metric = &mut ctx.syscall_table().metrics.borrow_mut()[SyscallKind::Keccak];
        metric.count += 1;

        // if full, the guest must call prove.
        Ok((self.is_full() as u32, 0))
    }

    fn keccak_prove(
        &mut self,
        ctx: &mut dyn SyscallContext,
        to_guest: &mut [u32],
    ) -> Result<(u32, u32)> {
        if !to_guest.is_empty() {
            bail!("invalid sys_keccak_prove");
        }

        let claim = ctx.load_digest_from_register(REG_A4)?;
        let control_root = ctx.load_digest_from_register(REG_A5)?;

        let rightsize = rightsize_enabled();
        let po2 = request_po2(self.inputs.len(), self.max_po2, rightsize);
        let proof_request = ProveKeccakRequest {
            claim_digest: claim,
            control_root,
            // Metal (K10) moves the inputs out; other builds clone them and clear after the
            // request is handed on, as upstream does (W10-GATE).
            input: if METAL_BUILD {
                std::mem::take(&mut self.inputs)
            } else {
                self.inputs.clone()
            },
            po2,
        };

        if let Some(coprocessor) = &ctx.syscall_table().coprocessor {
            coprocessor.borrow_mut().prove_keccak(proof_request)?;
        } else {
            ctx.syscall_table()
                .pending_keccaks
                .borrow_mut()
                .push(proof_request);
        }

        let metric = &mut ctx.syscall_table().metrics.borrow_mut()[SyscallKind::ProveKeccak];
        metric.count += 1;
        metric.size += 1 << po2 as u64;

        // reset
        self.inputs.clear();

        Ok((0, 0))
    }
}

#[cfg(test)]
mod tests {
    use risc0_circuit_keccak::{max_keccak_inputs, KECCAK_PO2_RANGE};

    use super::{request_po2, rightsize_from, KECCAK_RIGHTSIZE_DEFAULT};

    /// W10-GATE: right-sizing is on by default only in Metal builds; the knob opts in or out on
    /// every build.
    #[test]
    fn rightsize_default_is_metal_only() {
        let metal = cfg!(all(
            feature = "metal",
            any(target_os = "macos", target_os = "ios")
        ));
        assert_eq!(KECCAK_RIGHTSIZE_DEFAULT, metal);
        assert_eq!(rightsize_from(None), metal);
        assert!(!rightsize_from(Some("0")));
        assert!(!rightsize_from(Some(" 0 ")));
        assert!(rightsize_from(Some("1")));
    }

    /// W10-GATE: off Metal (knob unset) a request carries the session's max po2, as upstream
    /// release-3.1: `request_po2(.., rightsize = false)` is the max for every input count.
    #[test]
    fn default_request_po2_is_upstream_off_metal() {
        if KECCAK_RIGHTSIZE_DEFAULT {
            return;
        }
        for max_po2 in KECCAK_PO2_RANGE {
            for n in [0, 1, max_keccak_inputs(max_po2)] {
                assert_eq!(request_po2(n, max_po2, rightsize_from(None)), max_po2);
            }
        }
    }

    #[test]
    fn request_po2_is_smallest_that_fits() {
        // Off: always the max.
        assert_eq!(request_po2(0, 17, false), 17);
        assert_eq!(request_po2(1, 15, false), 15);
        // Empty and tiny batches: the smallest po2 of the range.
        assert_eq!(request_po2(0, 17, true), 14);
        assert_eq!(request_po2(1, 18, true), 14);
        for max_po2 in KECCAK_PO2_RANGE {
            for po2 in *KECCAK_PO2_RANGE.start()..=max_po2 {
                let full = max_keccak_inputs(po2);
                // A full batch of po2 stays at po2 ...
                assert_eq!(request_po2(full, max_po2, true), po2, "{max_po2} {po2}");
                // ... one more permutation needs the next po2 (or the max, its batch limit).
                let next = request_po2(full + 1, max_po2, true);
                assert_eq!(next, (po2 + 1).min(max_po2), "{max_po2} {po2}");
                // Never above the max, and always holds the inputs at or below the max's capacity.
                for n in [full / 2, full] {
                    let got = request_po2(n, max_po2, true);
                    assert!(got <= max_po2 && max_keccak_inputs(got) >= n);
                }
            }
        }
    }
}
