#!/usr/bin/env bash
# R17 audit: determinism stress of every Metal kernel family, plus the positive control and the
# static IR scan. Run after a Metal toolchain or macOS update (and before trusting a GPU phase):
#
#   source ~/zkminer/env.sh; export CARGO_BUILD_JOBS=6
#   tools/metal_audit/run_stress.sh <log dir> [reps]        # from the repo root
#
# Env: R17_AUDIT_SEGMENTS (comma list of rv32im segment postcards `(u32, Segment)`; without it the
# rv32im accum stress is skipped), R17_AUDIT_PO2 (zkp, default 20). Every stress line is
# `R17AUDIT <family> <case>: k/n reps differ from rep 0, m/n differ from CPU`; any nonzero k or m
# is a P0 (a nondeterministic kernel). The control line `R17AUDIT control repro: k/n` shows whether
# the Apple GPU defect is present (k = n so far; then the stress results say the other kernels do
# not hit it under this load). Summary at the end; exit status 1 if any stress test failed.
# R17_AUDIT_ONLY="control zkp ..." runs a subset (names: control zkp rv32im_accum rv32im_eval
# recursion keccak_eval keccak_witgen groth16 ir); the whole set takes ~8 min on an M5 Max.
set -uo pipefail
LOG=${1:?log dir}; REPS=${2:-30}; mkdir -p "$LOG"
ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd "$ROOT"
fail=0
run() { # name, env..., -- cargo args
  local name=$1; shift
  case " ${R17_AUDIT_ONLY:-$name} " in *" $name "*) ;; *) return 0 ;; esac
  echo "== $name" >&2
  env "$@" > "$LOG/$name.log" 2>&1 || { echo "   FAILED (see $LOG/$name.log)" >&2; fail=1; }
  grep -E 'R17AUDIT|R12 evalfast repeat|^po2 .*GPU witnesses|test result' "$LOG/$name.log" >&2
}
T="cargo test --features metal --lib"
run control   R17_AUDIT_REPS=10 $T -p risc0-zkp -- r17_defect_control --ignored --nocapture
run zkp       R17_AUDIT_REPS=$REPS $T -p risc0-zkp -- hal::metal::tests_r17 --ignored --nocapture --test-threads 1
if [ -n "${R17_AUDIT_SEGMENTS:-}" ]; then
  run rv32im_accum R17_AUDIT_REPS=$REPS $T -p risc0-circuit-rv32im -- r17_accum_segment_stress --ignored --nocapture
fi
run rv32im_eval R12_EF_PO2=20 R12_EF_REPS=$REPS R12_BASE_REPS=10 $T -p risc0-circuit-rv32im -- evalfast_repeat_bit_exact --ignored --nocapture
run recursion R17_AUDIT_REPS=$REPS R17_AUDIT_PO2=18 $T -p risc0-circuit-recursion -- r17_seal_repeat --ignored --nocapture
run keccak_eval R17_AUDIT_REPS=$REPS R17_AUDIT_PO2=17 $T -p risc0-circuit-keccak -- r17_eval_check_repeat --ignored --nocapture
run keccak_witgen K8_WITGEN_PO2S=17,18 K8_WITGEN_REPS=$REPS $T -p risc0-circuit-keccak -- metal_witgen_repeatable_long --ignored --nocapture
run groth16   R17_AUDIT_REPS=10 R17_AUDIT_LG=20 $T -p risc0-groth16-sys -- r17_ --ignored --nocapture --test-threads 1
# Static scan of the IR the build compiles (tiers: see ir_scan.py).
case " ${R17_AUDIT_ONLY:-ir} " in *" ir "*) ;; *) exit $fail ;; esac
tools/metal_audit/emit_ir.sh . target/debug "$LOG/ir" -gline-tables-only 2>/dev/null &&
  python3 -I tools/metal_audit/ir_scan.py --json "$LOG/ir_scan.json" --tiers '' "$LOG/ir" > "$LOG/ir_scan.txt" &&
  echo "== ir_scan: $LOG/ir_scan.txt" >&2 && awk 'NR==1 || $4+$5 > 0' "$LOG/ir_scan.txt" >&2
echo "== stress status: $([ $fail = 0 ] && echo all passed || echo FAILURES)" >&2
exit $fail
