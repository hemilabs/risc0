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
# not hit it under this load). `R17AUDIT z13 shape ...: k/n` (zkp step, `r17_shape_probe_z13`) probes
# the shape Z13 reported (a guarded element-wise update of a private array in fully inlined, unrolled
# code); it has never reproduced (0/n, W7-T1 fix round 1), so it is a stress case (k > 0 is a P0), not a
# positive control. Summary at the end; exit status 1 if any stress test failed.
# P31-T1: tier-Z bypass stress. The zkp step includes `r17_tierz_bypass` (eltwise_sum_fpext,
# eu2_reduce, batch_evaluate_any, max(REPS, 20) reps per shape); the `groth16_tierz` step runs
# groth16-sys `tierz_msm_window_bypass` (msm_window_reduce_{g1,g2,g1w,g2w}, max(REPS, 20) reps) and
# `tierz_g14_repeat` (R17_TIERZ_G14_REPS, default 50). Lines `R17AUDIT ... tierz ...`.
# R17_AUDIT_ONLY="control zkp ..." runs a subset (names: control zkp rv32im_accum rv32im_eval
# recursion keccak_eval keccak_witgen groth16 groth16_tierz ir); the whole set takes ~8 min on an
# M5 Max (plus the tier-Z steps). EMIT_IR_UNITS (see emit_ir.sh) limits the ir step to the units
# that build, e.g. EMIT_IR_UNITS="keccak zkp groth16" while rv32im/recursion Metal do not build.
# Run the whole script under one heavy `gpu-run` (compile with `cargo test --no-run` first).
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
run control   R17_AUDIT_REPS=10 $T -p risc0-zkp -- r17_defect_control --ignored --nocapture --test-threads 1
run zkp       R17_AUDIT_REPS=$REPS $T -p risc0-zkp -- hal::metal::tests_r17 --ignored --nocapture --test-threads 1
if [ -n "${R17_AUDIT_SEGMENTS:-}" ]; then
  run rv32im_accum R17_AUDIT_REPS=$REPS $T -p risc0-circuit-rv32im -- r17_accum_segment_stress --ignored --nocapture
fi
run rv32im_eval R12_EF_PO2=20 R12_EF_REPS=$REPS R12_BASE_REPS=10 $T -p risc0-circuit-rv32im -- evalfast_repeat_bit_exact --ignored --nocapture
run recursion R17_AUDIT_REPS=$REPS R17_AUDIT_PO2=18 $T -p risc0-circuit-recursion -- r17_seal_repeat --ignored --nocapture
run keccak_eval R17_AUDIT_REPS=$REPS R17_AUDIT_PO2=17 $T -p risc0-circuit-keccak -- r17_eval_check_repeat --ignored --nocapture
run keccak_witgen K8_WITGEN_PO2S=17,18 K8_WITGEN_REPS=$REPS $T -p risc0-circuit-keccak -- metal_witgen_repeatable_long --ignored --nocapture
run groth16   R17_AUDIT_REPS=10 R17_AUDIT_LG=20 $T -p risc0-groth16-sys -- r17_ --ignored --nocapture --test-threads 1
TZ=$(( REPS < 20 ? 20 : REPS ))
run groth16_tierz R17_TIERZ_REPS=$TZ R17_TIERZ_G14_REPS=${R17_TIERZ_G14_REPS:-50} $T -p risc0-groth16-sys -- tierz_ --nocapture --test-threads 1
# Static scan of the IR the build compiles (tiers: see ir_scan.py).
case " ${R17_AUDIT_ONLY:-ir} " in *" ir "*) ;; *) exit $fail ;; esac
# W7-T1: fail loudly (emit_ir.sh names the crate whose debug metal build is missing) instead of
# producing nothing. Tiers in R17_AUDIT_FAIL_TIERS (default AS) are gates (exit 1 if any is
# found). Tier Z (Z13's reported guarded-loop shape) is an UNVALIDATED heuristic: it is reported,
# not gated, by default (no GPU failure of the shape has been reproduced; notes/P31-T1.md).
echo "== ir" >&2
if ! tools/metal_audit/emit_ir.sh . target/debug "$LOG/ir" -gline-tables-only 2> "$LOG/emit_ir.log"; then
  sed 's/^/   /' "$LOG/emit_ir.log" >&2
  echo "   FAILED: emit_ir.sh (see $LOG/emit_ir.log); no IR scan was made" >&2; fail=1
else
  tail -1 "$LOG/emit_ir.log" >&2
  FT=${R17_AUDIT_FAIL_TIERS:-AS}
  python3 -I tools/metal_audit/ir_scan.py --json "$LOG/ir_scan.json" --tiers 'ASZ' --fail-tiers "$FT" "$LOG/ir" > "$LOG/ir_scan.txt"
  rc=$?
  echo "== ir_scan: $LOG/ir_scan.txt" >&2; awk 'NR==1 || ($1 ~ /\.ll$/ && $4+$5+$7+$8 > 0)' "$LOG/ir_scan.txt" >&2
  tot=$(grep '^ir_scan totals:' "$LOG/ir_scan.txt")
  [ -n "$tot" ] || { echo "   FAILED: ir_scan.txt has no totals line" >&2; fail=1; }
  echo "   $tot" >&2
  nz=$(sed -n 's/.* Z=\([0-9]*\).*/\1/p' <<< "$tot")
  [ "${nz:-0}" = 0 ] || echo "   note: $nz tier-Z finding(s) (unvalidated heuristic: guarded loop update of a private array; see ir_scan.txt)" >&2
  [ $rc = 0 ] || { echo "   FAILED: ir_scan found tier(s) $FT shapes" >&2; fail=1; }
fi
echo "== stress status: $([ $fail = 0 ] && echo all passed || echo FAILURES)" >&2
exit $fail
