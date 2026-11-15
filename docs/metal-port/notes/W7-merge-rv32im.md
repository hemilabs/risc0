# W7 merge record: unit rv32im (2026-11-15)

Merged into `zkminer/metal` with `--no-ff`, in this order:

- `metal/W7-R2` (966470a88, based on W7-A1 6af50cf84): po2 19 fits at 4 GiB and po2 20 at 8 GiB. It adds an
  in-place accum commit, an early drop of the data witness, a post-mix coefficient drop and a headroom-based
  prefill cap.
- `metal/W7-R1` (460f8615d): memory-safe defaults for the rv32im heavy tests, suite locks, and the guard's
  fallback order (`_nohoist` first).

The two items touch disjoint files, so there were no conflicts and no interaction fixes. Neither item changes
a Metal kernel source. Each branch is a single commit, and neither has a WIP commit. The open issues of both
items are listed in `W7-R2.md` and `W7-R1.md`.

## Gate (main checkout after both merges, logs in `scratch/w7/merge-rv32im/`)

`RISC0_METAL_ARCHIVE_GC=0` was set on every GPU run (shared machine, W7-Z1 note).

| command | result |
|---|---|
| `risc0-circuit-rv32im --features metal --lib`, default parallel harness, twice | 147 passed / 0 failed / 18 ignored, both runs (47-48 s) |
| R17 reproducer `witgen_segment_file`, 50 reps on `fixtures/R8/fix1/segs_torture20/seg0.postcard` | 50 / 50 good: 0 input rows changed, 0 output rows differ from the CPU |
| `seal_matches_cpu_large --ignored`, `R11_EQ_PO2S=`, the six R11 fix2 po2 20 segments (two `--huge` runs of three) | 6 of 6 seals equal the CPU seal and verify (CPU 53.6-54.6 s, Metal 1.14-1.26 s each); child peak 9.79-9.80 GiB |
| `run_stress.sh` `rv32im_eval` (po2 20, 30 reps) | 0 / 30 evalfast runs differ, 0 / 10 baseline runs differ |
| R22 regime guard (`eval_regime` filter: `eval_regime_guard_kernels` + 3 parser tests) | 4 passed; plain kernel in the fast regime (3984 B) |
| `cargo check -p risc0-zkvm --features metal` | ok |
| `risc0-zkp --features metal --lib hal::metal`, default parallel harness, twice | 201 passed / 0 failed / 48 ignored, both runs (76-77 s) |
| `risc0-circuit-rv32im-sys --features metal` (R17 guards) | 10 + 2 passed |
| `emit_ir.sh` + `ir_scan.py` | 175 TUs, tier A = S = 0 (B = 28, C = 29, unchanged from the zkp merge) |
