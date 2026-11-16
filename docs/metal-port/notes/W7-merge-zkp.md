# W7 merge record: unit zkp (2026-11-15)

Merged into `zkminer/metal` with `--no-ff`:

- `metal/W7-A1` (6af50cf84): process-wide pool cap, pressure drain, system-wide GPU check.
- `metal/W7-Z1` (d28ddca48): offline PSO archives for first install, fc specializations, archive GC.

The two items touch disjoint files, so there were no conflicts and no interaction fixes. Neither changes a
Metal kernel source.

Not merged: `metal/W7-T1` (6ee2bd1f9, status unresolved). Both remaining blocking findings were reproduced by
the reviewer and are not false positives:

1. `bin/memguard` STOP_MAX probation entry lowers `last_stop_wired` while the victim is still stopped. The
   reviewer's `t_drop.sh` shows an older flat process being stopped with no real growth. The one-line fix is
   known.
2. `ntt.rs` `split_tests::bad()` reads `RISC0_METAL_NTT_SPLIT_CHECK`. With the documented workaround
   `=0`, `known_bad_splits_are_refused` fails.

W7-T1 needs a fix round before it is merged.

## Gate (main checkout after both merges, logs in `scratch/w7/merge-zkp/`)

`RISC0_METAL_ARCHIVE_GC=0` was set on every GPU run (W7-Z1 note: shared machine).

| command | result |
|---|---|
| `risc0-zkp --features metal --lib hal::metal`, default parallel harness, twice | 198 passed / 0 failed / 48 ignored, both runs (70 s) |
| `risc0-zkp --features prove --lib` | 30 passed |
| `rv32im integration_tests::seal_matches_cpu_large --ignored`, `R11_EQ_PO2S=`, `R11_SEGMENTS` = the six R11 fix2 po2 20 segments (two runs of three) | 6 of 6 seals equal the CPU seal and verify (CPU 51-54 s, Metal 1.0-1.2 s each) |
| `risc0-circuit-recursion --features metal` | 77 + 1 passed, 19 ignored |
| `cargo check -p risc0-zkvm --features metal` | ok |
| `risc0-circuit-rv32im-sys --features metal` (R17 guards) | 10 + 2 passed |
| `emit_ir.sh` + `ir_scan.py` | 175 TUs, tier A = S = 0 (B = 28, C = 29) |
