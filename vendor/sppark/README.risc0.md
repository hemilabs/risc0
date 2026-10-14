# sppark 0.1.14, with four barriers added

This is the [sppark](https://github.com/supranational/sppark) 0.1.14 crate as published on
crates.io (`.crate` sha256 `d9c900139f3f6fdc8db217881a946adf00e935102fdd82b0e1bc19bacbffa311`,
the version this repository's `Cargo.lock` already used), unmodified except for four
`__syncthreads()` calls, each marked `risc0:`. `risc0-barriers.patch` is the whole diff.
sppark is Apache-2.0 (Supranational LLC); see `LICENSE`.

Each missing barrier is a shared-memory write-after-read race: a warp overwrites a shared
buffer that another warp of the same block may still be reading.

- `sppark/ntt/kernels/gs_mixed_radix_narrow.cu` (`_GS_NTT`, coalesced, `iterations > 6`): the
  last shared-memory stage reads another warp's row, then `transpose()` reuses the buffer.
- `sppark/polynomial/div_by_x_minus_z.cuh` (`d_div_by_x_minus_z`): three `xchg` reuses, at the
  top of each chunk, before the grid carry is stored, and before the `N > 1` carry step.

They corrupt real outputs, not just sanitizer reports. On an RTX 5090 with a shared-memory-heavy
kernel running concurrently, the stock GS NTT corrupted 32.3M elements in 400 2^20 transforms
and 8.9M in 2000 2^17 transforms; with the barriers, none. Under `compute-sanitizer` the stock
crate makes risc0 proofs fail verification, and racecheck reports the races; with the barriers
both are clean. `_CT_NTT` already ends its exchange loop with a barrier and needs none.

Upstream 0.1.15 has the same code. Drop this directory and go back to the crates.io release once
a release fixes them.

The workspace `Cargo.toml` points at it with `[patch.crates-io]`, which applies to this
repository's own builds only. A crate that depends on this repository resolves sppark through its
own lockfile and patches; zk-miner patches in the hemilabs/risc0-sppark fork, which needs the
same four barriers. The directory is excluded from the workspace and from `license-check.py`, so
its upstream code is not held to this repository's lints and headers.
