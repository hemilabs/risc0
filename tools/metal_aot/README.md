# Offline (AOT) Metal binary archives: build and packaging manifest

Z16 added `RISC0_METAL_AOT_DIR` (read-only offline archives made by `metal-tt`); W7-Z1 finished it
(OS key, function-constant specializations, seeding, the groth16/keccak hooks) and wrote these tools.
Runtime side: `risc0/zkp/src/hal/metal/archive.rs` (module docs). Measurements and the go/no-go
evidence: `docs/metal-port/notes/W7-Z1.md`. This file is the packaging manifest for W7-M11.

## What a first install saves (M5 Max, `BENCH_ONLY=fibonacci` worker, W7-Z1 note)

| | no offline archives | with them |
|---|---|---|
| first install (cold OS shader cache, empty runtime archive dir): pipelines | 73.0-89.7 s | 0.087-0.088 s |
| same, worker process wall | 75.0-91.7 s | 2.10-2.11 s |
| po2 20 specialized eval PSO (`ef_eval_split_fc`), first install | 18.3-24.0 s | 5.5-6.1 s (the rest is `newFunctionWithName:constantValues:`, which no archive removes) |

## Files

`<unit>-<digest24>-<arch>-<os key>-<checksum16>.binarchive`

- `unit`: the runtime archive unit (`zkp`, `rv32im`, `rv32im_fc<domain>`, `recursion`,
  `recursion_splitrg`, `keccak`, `groth16`; optional `keccakstrict` and the recursion revert variants).
- `digest24`: the first 24 hex digits of the parallel chunked sha256 (`archive::parallel_digest`) of the
  metallib the binary embeds. A different risc0 build (any kernel change) has another digest: rebuild.
- `arch`: `MTLDevice.architecture.name`, the `metal-tt -arch` name.
- `os key`: `macos<major>` (built with `-macos-version-min <major>.0`; used by every build of that major
  version) or `os<ProductBuildVersion>` (that build only). The runtime tries its exact build first, then its
  major version; any other key is never opened.
- `checksum16`: first 16 hex digits of the parallel digest of the file; the runtime checks it before Metal
  reads the file (a damaged archive can abort the process inside Metal).

Every file is optional: a missing or mismatched file is a clean miss and the PSO is compiled at runtime
(into the runtime archive) as without offline archives. When an offline archive serves every PSO of a unit
and there is no runtime archive yet, the runtime clones it into the runtime archive
(`~/Library/Caches/com.risczero.RISC-Zero.risc0/metal-archives`, APFS clone: no extra space), so the
offline directory can be removed after the first start.

## Architectures

`metal-tt` 32023.922 (Metal toolchain 27B47, `-macos-version-min 27.0`) builds these Mac GPU families:

| arch | Macs (Apple naming, not verified on hardware here) | built | validated on hardware |
|---|---|---|---|
| applegpu_g13g | M1 | yes | no |
| applegpu_g13s | M1 Pro | yes | no |
| applegpu_g13d | M1 Ultra | no (buildable) | no |
| applegpu_g13c | M1 Max | **no: "No available translator"** | - |
| applegpu_g14g | M2 | yes | no |
| applegpu_g14s | M2 Pro / Max | yes | no |
| applegpu_g14d | M2 Ultra | no (buildable) | no |
| applegpu_g15g | M3 | yes | no |
| applegpu_g15s | M3 Pro / Max | yes | no |
| applegpu_g15d | M3 Ultra | no (buildable) | no |
| applegpu_g16g | M4 | yes | no |
| applegpu_g16s | M4 Pro / Max | yes | no |
| applegpu_g17g | M5 | yes | no |
| applegpu_g17s | M5 Pro / Max (this Mac reports it) | yes | **yes** (W7-Z1: unit suites, seal equality, stress) |

**Only applegpu_g17s is validated.** The other files are `metal-tt` output that no GPU of that family has
run. Before shipping one, run on such a Mac, with `RISC0_METAL_AOT_DIR` set and an empty
`RISC0_METAL_ARCHIVE_DIR`: the rv32im `seal_matches_cpu_large` test on R11 po2 20 segments with
`RISC0_METAL_RV32IM_EVAL_FC=wait` (seals must equal the CPU's), the recursion, keccak and groth16-sys lib
suites, and `tools/metal_audit/run_stress.sh` for the rv32im_eval and recursion families. A GPU binary
from the offline compiler differs from the OS compiler's, so bit-exactness has to be shown per family.

## Build (no GPU needed beyond loading each metallib once)

```sh
source ~/zkminer/env.sh            # metal-tt on PATH
# 1. dump the exact embedded metallibs and kernel sets of this build (each crate's lib test binary):
for t in risc0_zkp:hal::metal::tests_z16 \
         risc0_circuit_rv32im:prove::hal::metal::eval_check::ef_tests \
         risc0_circuit_keccak:prove::hal::metal::archive::tests \
         risc0_groth16_sys:metal::pipelines::tests; do
  W7Z1_DUMP=$DUMP target/release/deps/${t%%:*}-<hash> --ignored --exact ${t#*:}::w7z1_dump_aot_inputs
done
# 2. build every unit for every arch (recursion from its build output and lib.rs):
python3 -I tools/metal_aot/build_all.py --dump $DUMP --out $OUT --jobs 5 \
  --recursion-metallib target/release/build/risc0-circuit-recursion-sys-*/out/metal_kernels_recursion.metallib \
  --recursion-sys-lib risc0/circuit/recursion-sys/src/lib.rs \
  --arch applegpu_g17s --arch applegpu_g16s ...
```

The metallibs must be the ones the shipped binary embeds: check that the `digest24` of the files equals
`archive::parallel_digest` of the worker's metallibs (`build_all.py` prints them in `manifest.jsonl`;
`metallib_sha256` there is the plain sha256 of the input metallib, comparable with
`target/release/build/*-sys-*/out/*.metallib` of the worker build).

Cost on an M5 Max (load 5-60, other agents running): about 4 min of 1 core per arch for the whole set
(rv32im 25-94 s, keccak 13-112 s, groth16 20-56 s, five fc units 20-44 s each, the rest a few s);
10 arches took 12 min wall with `--jobs 5` (~2.2 GB RAM per metal-tt for rv32im).

`metallib_extract.py`: `metal-tt` 32023.922 cannot specialize a function whose metadata is not the
first in its library ("cannot find private metadata at offset N"); `build_archive.py --fc` therefore
builds from a library holding only that function (byte-identical function, hash and metadata). The runtime
matches such an archive against the full library (W7-Z1 go/no-go).

## Packaging manifest (for W7-M11)

The set built by W7-Z1 for risc0 `metal/W7-Z1` (kernels identical to `zkminer/metal` b5fa9a97a, wave-6
worker) is in `/Users/user/zkminer/scratch/w7/W7-Z1/aot/` (not in git: 1.7 GB):

- 110 files = 11 units x 10 arches; `manifest.jsonl` (one JSON line per file: `file`, `unit`, `arch`,
  `os_key`, `macos_min`, `metallib_digest24`, `metallib_sha256`, `names`, `fc`, `bytes`, `sha256`,
  `metal_tt`, `build_seconds`) and `SHA256SUMS` (`shasum -a 256 -c SHA256SUMS`).
- Per arch 168-175 MB: rv32im 52-53 MB, keccak 36-38 MB, rv32im_fc x5 11.1-11.6 MB each, recursion
  7.4-7.8 + 6.9-7.1 MB, groth16 6.6-7.5 MB, zkp 3.3-3.8 MB.
- Metallib digests (digest24): zkp `03b74d3e3a530427f2f917e6`, rv32im (and fc units)
  `e6c3ee440733fd3286b448d0`, recursion `a36993cc7cbdfdcc422dc457`, keccak `0f98bbd23ec6976619e4b9b5`,
  groth16 `a99c12671ad7e657e98796b1`. Any risc0 kernel change makes them stale (clean misses): rebuild the
  set from the shipped binary's build tree.

What to ship and how:

1. Ship only the files of the user's arch (detect with `MTLDevice.architecture.name`, or ship all of one
   major macOS version and let the runtime pick). Download/unpack into a directory the worker can read,
   e.g. `~/.zkminer/metal-aot/<risc0 version>/`, verify `SHA256SUMS`, and start the worker with
   `RISC0_METAL_AOT_DIR=<that dir>`.
2. Shipping only g17s today is safe (validated). Ship another family only after the hardware validation
   above; until then those Macs compile at first start as now (65-90 s) and keep their runtime archives.
3. Units can be shipped selectively: rv32im (+zkp) is ~60 s of the first-install compile, groth16 ~30 s,
   recursion and keccak ~15-30 s each (in parallel); the fc units remove ~13 s per segment po2 on first use.
4. A new macOS major version needs a rebuild with `--macos-min <major>.0` (the files of the previous major
   are not opened). A minor update keeps using `macos<major>` files; if Metal finds a binary unusable it is
   a miss (compile), never a wrong PSO.
5. After the first successful start the offline dir can be deleted (the runtime archive holds the clones).

Knobs: `RISC0_METAL_AOT_DIR` (unset: no offline archives), `RISC0_METAL_AOT_FC=0` (no specialized PSOs from
them), `RISC0_METAL_AOT_SEED=0` (no clone into the runtime archive), `RISC0_METAL_ARCHIVE=0` (no archives
at all).
