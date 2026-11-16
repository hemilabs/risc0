# stock_verify31

A receipt verifier built only from the released crates.io risc0 3.1 crates (`risc0-zkvm =3.1.0`,
`risc0-circuit-recursion =4.1.0`, `risc0-circuit-rv32im =4.1.0`, default features off). It judges the
receipts this tree's Metal prover makes, so that the tree's own verifier is not the only judge
(item P31-14, `docs/metal-port/notes/P31-14.md`).

- Not a member of the risc0 workspace: it has its own `[workspace]`, no path dependency and no
  `[patch]`. `build.rs` fails the build unless `Cargo.lock` lists every `risc0-*` / `bonsai-*`
  package with the crates.io registry source and `risc0-zkvm` at 3.1.0, the manifest has no path or
  git dependency, `[patch` or `[replace`, and no cargo config file cargo reads for the build (the
  crate dir and its ancestors, `CARGO_HOME`) has a `paths` override or a `[patch`. The lock is
  committed and required; `run_matrix.sh` builds with `--locked`.
- Receipt files are `risc0_zkvm::Receipt` encoded with bincode 1.3, the encoding the 3.1 crate uses
  for receipts itself (`api` client/server, `Asset::Inline`).

```
stock-verify31 verify <receipt> <image id hex> [--expect-journal <hex> | --expect-journal-file <path>] [--relabel-as-31]
stock-verify31 consts
```

`verify` prints what the receipt carries (kind, rv32im seal version word, control ids, verifier
parameter digests, the Groth16 selector), then runs the stock `Receipt::verify(image_id)` with the
default 3.1 `VerifierContext`. It exits 0 when the receipt verifies, 1 when it is rejected (the error
is printed) and 2 on a usage or I/O error. `--relabel-as-31` is for negative tests: it first sets
every version label (seal version 3, segment / composite / succinct / Groth16 parameter digests) to
the 3.1 value, so that a receipt from another version is judged by the cryptographic checks and not
only by its labels. `consts` prints the stock 3.1 constants.

## Producer (`produce/`)

`p31-14-produce` proves with the tree it is built against (`--features metal`; its path
dependencies point at this tree) and writes the composite, succinct and Groth16 receipts of one
case, verifying each with that tree's verifier first:

- `fibonacci` and `merkle`: zk-miner's benchmark guests (ProgramBinary `.bin`), with the worker's
  `--benchmark` inputs (1000 and 512, as 4 LE bytes);
- `compose`: multi_test `SysVerify` of two proven `Echo` receipts, so the succinct receipt goes
  through resolve;
- `keccak`: multi_test `KeccakUnion(1)`, a keccak assumption.

`wrap-timing <elf> <input> <reps>` makes one succinct receipt and wraps it `<reps>` times in one
process (the identity_p254 / shrink_wrap timing of P31-14). Its `Cargo.lock` is not committed: it
follows the tree it is built against. Seed it with the tree's `Cargo.lock` before a manual build
(`run_matrix.sh` does); a fresh resolve picks newer crates that fail the tree's `-Dwarnings`.

It prints the device allocation each stage leaves and, before the wrap, the headroom the Groth16 key
load will see. Diagnostics: `P14_TRACE_ALLOC=1` (every allocation change >= 1 MiB),
`P14_DIAG_IDENTITY=1` (identity_p254 alone first), `P14_PREBUILD=keccak|groth16|rv32im|recursion|worker`
(build those process-wide PSO sets at start-up; `worker` builds all four as zk-miner's worker does).

## Matrix

`run_matrix.sh` builds everything into `OUT`, makes the receipts with this tree and, with
`FROZEN30=<3.0 tree>`, with a copy of the producer pointed at that tree (nothing is written into
either tree), re-proves memory-merkle, keccak and fibonacci-after-the-worker's-PSO-warmup at the 8
and 4 GiB budgets, and prints the matrix. It exits 1 unless every 3.1 receipt verifies and every 3.0
receipt is rejected (as made and relabelled). `KNOWN_FAIL` lists runs whose Groth16 wrap is known to
be refused by the budget (P31-14 open issue 1): they must fail exactly there, and the script exits 1
if one starts passing. Every prove runs under `gpu-run` (heavy). `CASES`, `VERSIONS` and `BUDGETS`
split a long run.

```
OUT=~/zkminer/scratch/w8/P31-14 GUESTS=<dir with fibonacci.bin, memory-merkle.bin> \
  FROZEN30=~/zkminer/deps/risc0 tools/stock_verify31/run_matrix.sh
```

The script sets `RISC0_GROTH16_DIR` per tree: `~/.risc0/extensions/v0.2.0-risc0-groth16` for 3.1 and
`v0.1.0-risc0-groth16` for 3.0.
