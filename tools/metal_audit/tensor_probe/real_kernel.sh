#!/usr/bin/env bash
# N1 step 3: the real ef_eval_split with its Fp poly_mix chains rewritten by mm_chain.py, timed and
# checked by the rv32im `ef_variant_bench` (release lib test binary) against the built-in kernel.
#   real_kernel.sh <out dir> [po2] [rounds]
# Needs `cargo test --release -p risc0-circuit-rv32im --features metal --lib --no-run` in this
# worktree (its OUT_DIR has the generated TU and headers). Variants (see notes/N1.md):
#   nlall   Fp links of chains with >= 8 Fp links -> xor (wrong results): their arithmetic share
#   tcm     tensor form, Q tile built in the kernel per block (bit-exact)
#   tcnoq   ... without the Q build (wrong results): staging + matmul + combine
#   tcnomm  ... without the matmul: staging + combine + barriers
#   acc1    chain-level accumulation in the cooperative tensor, Q built in the kernel (bit-exact)
#   acc2    acc1 without the Q build (wrong results): the best case of the tensor form
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); WT=$(cd "$D/../../.." && pwd); OUT=${1:?out dir}; PO2=${2:-20}; R=${3:-7}
mkdir -p "$OUT"
SYS=$(ls -td "$WT"/target/release/build/risc0-circuit-rv32im-sys-*/out | while read d; do [ -f "$d/metal/eval_check_ef.metal" ] && { echo "$d"; break; }; done)
G=$SYS/metal/eval_check_ef.metal
python3 "$D/mm_chain.py" "$G" "$OUT/all_nolink.metal" --funcs all --mode nolink
python3 "$D/mm_chain.py" "$G" "$OUT/all_tensor.metal" --funcs all
python3 "$D/mm_chain.py" "$G" "$OUT/all_acc.metal" --funcs all --acc
I=(-I "$SYS/_sys_" -I "$WT/risc0/circuit/rv32im-sys/kernels/metal" -I "$SYS/metal")
mk() { local name=$1 src=$2; shift 2
  xcrun -sdk macosx metal -std=metal4.0 -O2 -Wno-unused-variable "${I[@]}" "$@" -c "$src" -o "$OUT/$name.air"
  xcrun -sdk macosx metallib "$OUT/$name.air" -o "$OUT/$name.metallib"; }
mk nlall "$OUT/all_nolink.metal" &
mk tcm "$OUT/all_tensor.metal" -DN1_MPP -DN1_TC -DN1_QMODE=1 &
mk tcnoq "$OUT/all_tensor.metal" -DN1_MPP -DN1_TC -DN1_QMODE=2 &
mk tcnomm "$OUT/all_tensor.metal" -DN1_MPP -DN1_TC -DN1_QMODE=2 -DN1_NOMM &
mk acc1 "$OUT/all_acc.metal" -DN1_MPP -DN1_TC -DN1_QMODE=1 &
mk acc2 "$OUT/all_acc.metal" -DN1_MPP -DN1_TC -DN1_QMODE=2 &
wait
B=$(ls -t "$WT"/target/release/deps/risc0_circuit_rv32im-* | grep -v '\.d$' | head -1)
V="A=builtin"; for v in nlall tcm tcnoq tcnomm acc1 acc2; do V="$V,$v=$OUT/$v.metallib"; done; V="$V,A2=builtin"
# The bench fails (exit 101) because the ablations differ from A by design; read `mismatching runs`.
R15E_PO2=$PO2 R15E_PAIRS=$R R15E_VARIANTS=$V "$B" prove::hal::metal::eval_check::ef_tests::ef_variant_bench \
  --ignored --nocapture 2>&1 | grep -E "R15E .*(split|load)" || true
