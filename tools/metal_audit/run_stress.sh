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
# R17_AUDIT_ONLY="control zkp ..." runs a subset (names: control zkp rv32im_gen rv32im_accum
# rv32im_witgen rv32im_amp rv32im_eval rv32im_eval_forms recursion recursion_<k> keccak_eval
# keccak_witgen groth16 groth16_tierz ir; recursion_<k> for k in R17_AUDIT_REC_KERNELS, default
# "lz lz32 split monolith", runs r17_seal_repeat with RISC0_METAL_REC_KERNELS=k); the whole set takes ~8 min on an
# M5 Max (plus the tier-Z steps). EMIT_IR_UNITS (see emit_ir.sh) limits the ir step to the units
# that build, e.g. EMIT_IR_UNITS="keccak zkp groth16" while rv32im/recursion Metal do not build.
# Run the whole script under one heavy `gpu-run` (compile with `cargo test --no-run` first).
# P31-16 (3.1, all R17/R17b/R18 gates in one place):
# - `rv32im_gen`: rv32im-sys generator tests (witgen_tu_gen.py --scan-loops / --scan-ir /
#   --scan-metallib on the built TU and metallib, ir_scan tiers ABS at six witgen settings, pins).
# - `rv32im_witgen`: witgen_segment_file on every R17_AUDIT_SEGMENTS segment, par
#   (R17_AUDIT_WITGEN_REPS calls each, default 200: 5 segments = R17's 0/1000 gate) and sorted
#   (R17_AUDIT_WITGEN_SORTED_REPS, default 30). Lines `R17AUDIT rv32im witgen <seg> <mode>: k/n`.
# - `rv32im_amp`: the sorted amplifier (`witgen_sorted_amplifier`, REPS calls) and sorted
#   witgen_segment_file on the first R17_AUDIT_AMP_SEGS (default 2) segments, for every
#   codegen setting in R17_AUDIT_AMP_LEVELS (default "O2 O0 Oz Os O1 auto auto_Oz auto_O0"; O2
#   = the embedded metallib, the others are witgen_tu built at that -O level with
#   build_rv32im_variant.sh, auto* = compiler-decided inlining, -DWITGEN_TU_FNATTR=), loaded
#   with RISC0_METAL_RV32IM_METALLIB. R17's perturbation gate. Libraries under $LOG/amp_libs.
# - `ir`: tiers A, S and P (R18's pair-load miscompile shape, ir_scan.py) are gates
#   (R17_AUDIT_FAIL_TIERS, default ASP); R17_AUDIT_IR_LEVELS (default "O2") adds -O levels
#   (each scanned into $LOG/ir_<L>; there only R17_AUDIT_FAIL_TIERS_X, default P, gates, since
#   only -O2 is shipped and witgen's six settings are gated by `rv32im_gen`); IR_SCAN_JOBS (default 4) scans files in parallel; the
#   tier-P positive control (R18's `r18_pmnld`: n loaded with a second pair load, from the
#   build's eval_check_ef.metal) must be flagged, else the step fails.
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
want() { case " ${R17_AUDIT_ONLY:-$1} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
# P31-16: witgen_segment_file prints one `[r8] seg ...` line per call; summarise as R17AUDIT lines.
wsum() { # log, label
  awk -v lab="$2" '/^\[r8\] seg po2=/ { n++; if ($0 !~ /output rows differing from cpu: 0 / || $0 !~ /: Ok\(/) k++ }
    END { printf "R17AUDIT rv32im witgen %s: %d/%d calls bad\n", lab, k, n; if (n == 0) exit 1 }' "$1"
}
RV_OUT() { ls -td target/debug/build/risc0-circuit-rv32im-sys-*/out 2>/dev/null | while read d; do if [ -d "$d/metal" ]; then echo "$d"; break; fi; done; }
run rv32im_gen cargo test --features metal -p risc0-circuit-rv32im-sys
run control   R17_AUDIT_REPS=10 $T -p risc0-zkp -- r17_defect_control --ignored --nocapture --test-threads 1
run zkp       R17_AUDIT_REPS=$REPS $T -p risc0-zkp -- hal::metal::tests_r17 --ignored --nocapture --test-threads 1
if [ -n "${R17_AUDIT_SEGMENTS:-}" ]; then
  run rv32im_accum R17_AUDIT_REPS=$REPS $T -p risc0-circuit-rv32im -- r17_accum_segment_stress --ignored --nocapture
fi
if [ -n "${R17_AUDIT_SEGMENTS:-}" ] && want rv32im_witgen; then
  echo "== rv32im_witgen" >&2
  IFS=, read -r -a segs <<< "$R17_AUDIT_SEGMENTS"
  for seg in "${segs[@]}"; do
    [ -n "$seg" ] || continue
    b=$(basename "$(dirname "$seg")")_$(basename "${seg%.postcard}")
    for mode in par sorted; do
      if [ $mode = par ]; then n=${R17_AUDIT_WITGEN_REPS:-200}; perm=; else n=${R17_AUDIT_WITGEN_SORTED_REPS:-30}; perm=sorted; fi
      lg="$LOG/rv32im_witgen_${b}_$mode.log"
      env RISC0_METAL_WITGEN_SEGMENT="$seg" RISC0_METAL_WITGEN_REPS=$n RISC0_METAL_WITGEN_MODES=par \
        RISC0_METAL_WITGEN_PERM=$perm $T -p risc0-circuit-rv32im -- witgen_segment_file --nocapture --test-threads 1 \
        > "$lg" 2>&1 || { echo "   FAILED (see $lg)" >&2; fail=1; }
      wsum "$lg" "$b $mode" >&2 || { echo "   FAILED: no witgen calls in $lg" >&2; fail=1; }
    done
  done
fi
if want rv32im_amp; then
  echo "== rv32im_amp" >&2
  OD=$(RV_OUT)
  if [ -z "$OD" ]; then echo "   FAILED: no rv32im-sys metal OUT_DIR under target/debug" >&2; fail=1; else
  IFS=, read -r -a asegs <<< "${R17_AUDIT_SEGMENTS:-}"
  for L in ${R17_AUDIT_AMP_LEVELS:-O2 O0 Oz Os O1 auto auto_Oz auto_O0}; do
    lib=; envlib=()
    if [ "$L" != O2 ]; then
      case $L in auto) fl="-DWITGEN_TU_FNATTR=" ;; auto_*) fl="-${L#auto_} -DWITGEN_TU_FNATTR=" ;; *) fl="-$L" ;; esac
      lib="$LOG/amp_libs/$L"
      if ! FLAGS_witgen_tu="$fl" tools/metal_audit/build_rv32im_variant.sh "$OD" "$lib" > "$LOG/amp_build_$L.log" 2>&1; then
        echo "   FAILED: variant build $L (see $LOG/amp_build_$L.log)" >&2; fail=1; continue
      fi
      envlib=(RISC0_METAL_RV32IM_METALLIB="$lib/rv32im.metallib")
    fi
    lg="$LOG/rv32im_amp_$L.log"
    env ${envlib[@]+"${envlib[@]}"} RISC0_METAL_WITGEN_AMP_REPS=$REPS $T -p risc0-circuit-rv32im -- witgen_sorted_amplifier --nocapture --test-threads 1 \
      > "$lg" 2>&1 || { echo "   FAILED (see $lg)" >&2; fail=1; }
    sed -n 's/^\[r17\] witgen_sorted_amplifier: /R17AUDIT rv32im amp '"$L"': /p' "$lg" >&2
    grep -q '^\[r17\] witgen_sorted_amplifier: ' "$lg" || { echo "   FAILED: no amplifier line in $lg" >&2; fail=1; }
    i=0
    for seg in ${asegs[@]+"${asegs[@]}"}; do
      [ -n "$seg" ] || continue; i=$((i+1)); [ $i -le ${R17_AUDIT_AMP_SEGS:-2} ] || break
      b=$(basename "$(dirname "$seg")")_$(basename "${seg%.postcard}")
      lg="$LOG/rv32im_amp_${L}_$b.log"
      env ${envlib[@]+"${envlib[@]}"} RISC0_METAL_WITGEN_SEGMENT="$seg" RISC0_METAL_WITGEN_REPS=$REPS RISC0_METAL_WITGEN_MODES=par \
        RISC0_METAL_WITGEN_PERM=sorted $T -p risc0-circuit-rv32im -- witgen_segment_file --nocapture --test-threads 1 \
        > "$lg" 2>&1 || { echo "   FAILED (see $lg)" >&2; fail=1; }
      wsum "$lg" "$L $b sorted" >&2 || { echo "   FAILED: no witgen calls in $lg" >&2; fail=1; }
    done
  done
  fi
fi
run rv32im_eval R12_EF_PO2=20 R12_EF_REPS=$REPS R12_BASE_REPS=10 $T -p risc0-circuit-rv32im -- evalfast_repeat_bit_exact --ignored --nocapture
# P31-16: every evalfast form (flat, fc, nested, fc_nest, unhoisted x2, r12, r15e, baseline) vs CPU
run rv32im_eval_forms $T -p risc0-circuit-rv32im -- evalfast_matches_cpu_po2_16 --nocapture --test-threads 1
run recursion R17_AUDIT_REPS=$REPS R17_AUDIT_PO2=18 $T -p risc0-circuit-recursion -- r17_seal_repeat --ignored --nocapture
# P31-16: the other recursion eval_check forms (RISC0_METAL_REC_KERNELS; the default is rg).
for k in ${R17_AUDIT_REC_KERNELS:-lz lz32 split monolith}; do
  run recursion_$k RISC0_METAL_REC_KERNELS=$k R17_AUDIT_REPS=$REPS R17_AUDIT_PO2=18 $T -p risc0-circuit-recursion -- r17_seal_repeat --ignored --nocapture
done
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
FT=${R17_AUDIT_FAIL_TIERS:-ASP}
export IR_SCAN_JOBS=${IR_SCAN_JOBS:-4}
for L in ${R17_AUDIT_IR_LEVELS:-O2}; do
  if [ "$L" = O2 ]; then d="$LOG/ir"; x=(); else d="$LOG/ir_$L"; x=(-$L); fi
  if ! tools/metal_audit/emit_ir.sh . target/debug "$d" -gline-tables-only ${x[@]+"${x[@]}"} 2> "$d.emit.log"; then
    sed 's/^/   /' "$d.emit.log" >&2
    echo "   FAILED: emit_ir.sh -$L (see $d.emit.log); no IR scan was made" >&2; fail=1; continue
  fi
  echo "   -$L: $(tail -1 "$d.emit.log")" >&2
  # Only -O2 is shipped: at other levels tiers A/S are reported (3.0 and 3.1 have the same latent
  # -Oz/-Os/-O0 sites outside witgen, notes/P31-16.md) and R17_AUDIT_FAIL_TIERS_X (default P) gates.
  if [ "$L" = O2 ]; then ft=$FT; else ft=${R17_AUDIT_FAIL_TIERS_X:-P}; fi
  python3 -I tools/metal_audit/ir_scan.py --json "$d.scan.json" --tiers 'ASZP' --fail-tiers "$ft" "$d" > "$d.scan.txt"
  rc=$?
  # (3.0 names, kept for the -O2 scan)
  [ "$L" = O2 ] && cp "$d.scan.txt" "$LOG/ir_scan.txt" && cp "$d.scan.json" "$LOG/ir_scan.json"
  echo "== ir_scan -$L: $d.scan.txt" >&2; awk 'NR==1 || ($1 ~ /\.ll$/ && $4+$5+$7+$8+$9 > 0)' "$d.scan.txt" >&2
  tot=$(grep '^ir_scan totals:' "$d.scan.txt")
  [ -n "$tot" ] || { echo "   FAILED: $d.scan.txt has no totals line" >&2; fail=1; }
  echo "   -$L $tot" >&2
  nz=$(sed -n 's/.* Z=\([0-9]*\).*/\1/p' <<< "$tot")
  [ "${nz:-0}" = 0 ] || echo "   note: $nz tier-Z finding(s) (unvalidated heuristic: guarded loop update of a private array; see $d.scan.txt)" >&2
  [ $rc = 0 ] || { echo "   FAILED: ir_scan -$L found tier(s) $ft shapes" >&2; fail=1; }
done
# P31-16: tier-P positive control (R18 section 4's r18_pmnld: the NBETA-folded row n loaded with
# a second 32-byte pair load instead of computed from b). The scan must flag it.
OD=$(RV_OUT)
if [ -n "$OD" ] && [ -e "$OD/metal/eval_check_ef.metal" ]; then
  c="$LOG/ir_controls"; mkdir -p "$c"
  sed 's/^#define EF_PMN_ROW(ctx, b, k) (ef_nbeta(b))$/#define EF_PMN_ROW(ctx, b, k) (EF_ROW((ctx).pm, EF_NPM + (k)))/' \
    "$OD/metal/eval_check_ef.metal" > "$c/eval_check_ef_pmnld.metal"
  if cmp -s "$OD/metal/eval_check_ef.metal" "$c/eval_check_ef_pmnld.metal"; then
    echo "   FAILED: tier-P control: EF_PMN_ROW default not found in eval_check_ef.metal (update run_stress.sh)" >&2; fail=1
  elif ! xcrun -sdk macosx metal -std=metal3.2 -O2 -Wno-unused-variable -gline-tables-only -S -emit-llvm \
      -c "$c/eval_check_ef_pmnld.metal" -o "$c/eval_check_ef_pmnld.ll" -I "$OD/_sys_" \
      -I risc0/circuit/rv32im-sys/kernels/metal -I "$OD/metal" 2> "$c/pmnld.err"; then
    echo "   FAILED: tier-P control did not compile (see $c/pmnld.err)" >&2; fail=1
  else
    python3 -I tools/metal_audit/ir_scan.py --tiers '' "$c/eval_check_ef_pmnld.ll" > "$c/scan.txt"
    np=$(sed -n 's/^ir_scan totals:.* P=\([0-9]*\).*/\1/p' "$c/scan.txt")
    echo "R17AUDIT ir tier-P control r18_pmnld: P=${np:-?} (must be > 0)" >&2
    [ "${np:-0}" -gt 0 ] || { echo "   FAILED: tier P did not flag the r18_pmnld control" >&2; fail=1; }
  fi
else
  echo "   note: no rv32im metal OUT_DIR, tier-P control skipped" >&2
fi
echo "== stress status: $([ $fail = 0 ] && echo all passed || echo FAILURES)" >&2
exit $fail
