#!/bin/zsh
# P31-14 receipt matrix: make composite / succinct / Groth16 receipts with this tree's Metal prover
# (and, with FROZEN30=<3.0 tree>, with that tree's), then judge each one with stock-verify31 (the
# crates.io risc0-zkvm 3.1.0 verifier). Nothing is written inside either risc0 tree: the builds use
# OUT/target-*, the 3.0 producer is a copy under OUT/prod30.
#
#   OUT=<scratch dir> GUESTS=<dir with fibonacci.bin, memory-merkle.bin> [FROZEN30=<3.0 tree>] \
#     [GPU_RUN=<gpu-run path>] [BUDGETS="8589934592 4294967296"] [KNOWN_FAIL="..."] \
#     tools/stock_verify31/run_matrix.sh
#
# GUESTS: zk-miner's benchmark guests (ProgramBinary .bin), proved with the worker's --benchmark
# inputs (fibonacci 1000, memory-merkle 512). Each prove runs under GPU_RUN (default
# ~/zkminer/bin/gpu-run, heavy lock). BUDGETS re-proves, with each producer, at those simulated
# working-set budgets (gpu-run --budget): memory-merkle, keccak (a coprocessor receipt), and
# fibonacci after the zk-miner worker's startup pipeline warmup (P14_PREBUILD=worker: rv32im,
# recursion, keccak and Groth16 PSOs built first, as `warm_up_pipelines` does; receipts in
# r<ver>_b<budget>_wpso).
# Expected: every 3.1 receipt "STOCK 3.1 VERIFY: OK", every 3.0 receipt "REJECTED" (also with
# --relabel-as-31). KNOWN_FAIL lists <receipt dir>/<case> runs whose Groth16 wrap is known to be
# refused by the working-set budget (open issue in docs/metal-port/notes/P31-14.md): for those the
# producer must fail with "Metal working-set budget exceeded" after writing composite and succinct
# receipts (which must still verify), and the matrix shows KNOWN-FAIL; a known failure that now
# passes is reported as an unexpected pass so the list gets updated. Exit 1 if any expectation
# fails.
# Long runs can be split (the matrix at the end judges every receipt already in OUT): CASES (default
# "fibonacci merkle compose keccak") picks the unbudgeted cases, VERSIONS (default "31 30") the
# producers to run, BUDGETS the budget runs ("" for none).
set -eu
HERE=${0:A:h}
TREE=${HERE:h:h}
: ${OUT:?set OUT} ${GUESTS:?set GUESTS}
GPU_RUN=${GPU_RUN:-$HOME/zkminer/bin/gpu-run}
BUDGETS=${BUDGETS-"8589934592 4294967296"}
# The 3.1 Groth16 key (v0.2.0) reserves 4039 MiB, leaving 57 MiB at a 4 GiB budget for everything
# else; the process-wide pipeline objects after a keccak proof, or after the worker's warmup, are
# 62 MiB, so the key load is refused (3.0's smaller key fits). See the P31-14 note.
KNOWN_FAIL=${KNOWN_FAIL-"r31_b4294967296/keccak r31_b4294967296_wpso/fibonacci"}
mkdir -p $OUT/logs
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-6}

CASES=${CASES-"fibonacci merkle compose keccak"}
VERSIONS=${VERSIONS-"31 30"}

# 1. The stock verifier, from the committed Cargo.lock (build.rs refuses non-registry risc0
# packages, path/git dependencies and cargo `paths`/[patch] overrides).
(cd $HERE && CARGO_TARGET_DIR=$OUT/target-stock cargo build --release --locked -q)
V=$OUT/target-stock/release/stock-verify31
if grep -A3 '^name = "risc0' $HERE/Cargo.lock | grep '^source' | grep -v 'registry+https://github.com/rust-lang/crates.io-index'; then
  echo "a risc0 package in tools/stock_verify31/Cargo.lock is not from crates.io" >&2; exit 1
fi
$V consts | tee $OUT/logs/stock_consts.log

# 2. Producers (outside the GPU lock). The producer's Cargo.lock is not committed; seed it from the
# tree's so it resolves the tree's pinned versions (a fresh resolve picks newer crates, e.g. a `cc`
# whose deprecations fail risc0-build-kernel under the tree's -Dwarnings).
[[ -f $HERE/produce/Cargo.lock ]] || cp $TREE/Cargo.lock $HERE/produce/Cargo.lock
(cd $HERE/produce && CARGO_TARGET_DIR=$OUT/target-prod31 cargo build --release -q)
PRODS=(31:$OUT/target-prod31/release/p31-14-produce:v0.2.0)
if [[ -n ${FROZEN30:-} ]]; then
  mkdir -p $OUT/prod30
  cp -R $HERE/produce/src $OUT/prod30/
  cp $FROZEN30/rust-toolchain.toml $OUT/prod30/
  [[ -f $OUT/prod30/Cargo.lock ]] || cp $FROZEN30/Cargo.lock $OUT/prod30/
  sed "s#path = \"../../../risc0/#path = \"${FROZEN30:A}/risc0/#" $HERE/produce/Cargo.toml > $OUT/prod30/Cargo.toml
  (cd $OUT/prod30 && CARGO_TARGET_DIR=$OUT/target-prod30 cargo build --release -q)
  PRODS+=(30:$OUT/target-prod30/release/p31-14-produce:v0.1.0)
fi

run() { # <log> <gpu-run args...> -- <cmd...>; returns the command's exit status
  local log=$1 rc; shift
  : > $log
  while :; do
    rc=0; $GPU_RUN "$@" </dev/null >>$log 2>&1 || rc=$?
    [[ $rc -ne 75 ]] && return $rc
    sleep 20
  done
}

fail=0
known() { [[ " $KNOWN_FAIL " == *" $1 "* ]]; }
# <ver> <receipt dir> <case> <log> <gpu-run args...> -- <cmd...>: one producer run, with the
# known-failure contract checked.
produce() {
  local ver=$1 dir=$2 c=$3 log=$4 rc=0; shift 4
  mkdir -p $OUT/receipts/$dir
  rm -f $OUT/receipts/$dir/$c.*(N)
  run $log "$@" || rc=$?
  if known $dir/$c; then
    if [[ $rc -eq 0 ]]; then
      echo "UNEXPECTED PASS: $dir/$c is in KNOWN_FAIL but its wrap succeeded (update the list)" >&2
      fail=1
    elif ! grep -q 'Metal working-set budget exceeded' $log \
        || [[ ! -f $OUT/receipts/$dir/$c.succinct.receipt ]]; then
      echo "$dir/$c failed other than at the known Groth16 key load (see $log)" >&2
      fail=1
    fi
  elif [[ $rc -ne 0 ]]; then
    echo "producer failed: $dir/$c exit $rc (see $log)" >&2
    fail=1
  fi
}

# 3. Receipts.
for p in $PRODS; do
  ver=${p%%:*}; rest=${p#*:}; bin=${rest%:*}; g16=${rest##*:}
  [[ " $VERSIONS " == *" $ver "* ]] || continue
  export RISC0_GROTH16_DIR=$HOME/.risc0/extensions/$g16-risc0-groth16
  for c in ${=CASES}; do
    case $c in fibonacci) A=($GUESTS/fibonacci.bin 1000);; merkle) A=($GUESTS/memory-merkle.bin 512);; *) A=();; esac
    produce $ver r$ver $c $OUT/logs/prod${ver}_$c.log --tag P31-14-prod$ver -- \
      $bin receipts $OUT/receipts/r$ver $c $A
  done
  for b in ${=BUDGETS}; do
    for c in merkle keccak; do
      case $c in merkle) A=($GUESTS/memory-merkle.bin 512);; *) A=();; esac
      produce $ver r${ver}_b$b $c $OUT/logs/prod${ver}_${c}_b$b.log --tag P31-14-prod$ver --budget $b -- \
        $bin receipts $OUT/receipts/r${ver}_b$b $c $A
    done
    produce $ver r${ver}_b${b}_wpso fibonacci $OUT/logs/prod${ver}_fibonacci_b${b}_wpso.log \
      --tag P31-14-prod$ver --budget $b -- \
      env P14_PREBUILD=worker $bin receipts $OUT/receipts/r${ver}_b${b}_wpso fibonacci $GUESTS/fibonacci.bin 1000
  done
done

# 4. The matrix.
printf "%-26s %-10s %-10s %-9s %s\n" made-by case kind expect result | tee $OUT/logs/matrix.txt
for d in $OUT/receipts/r*(/); do
  name=${d:t}; expect=OK; [[ $name = r30* ]] && expect=REJECTED
  for idf in $d/*.image_id; do
    c=${idf:t:r}
    for k in composite succinct groth16; do
      for relabel in "" --relabel-as-31; do
        [[ -n $relabel && $expect = OK ]] && continue
        if [[ ! -f $d/$c.$k.receipt ]]; then
          if [[ $k = groth16 ]] && known $name/$c; then
            line="KNOWN-FAIL: no receipt, the Groth16 key load was refused by the working-set budget"
          else
            line="MISSING: no receipt"; fail=1
          fi
          printf "%-26s %-10s %-10s %-9s %s\n" "$name${relabel:+ relabel}" $c $k $expect "$line" | tee -a $OUT/logs/matrix.txt
          continue
        fi
        line=$($V verify $d/$c.$k.receipt $(cat $idf) --expect-journal "$(cat $d/$c.journal)" $relabel | grep '^STOCK' || true)
        got=${${line#STOCK 3.1 VERIFY: }%% *}
        [[ $got = $expect ]] || fail=1
        printf "%-26s %-10s %-10s %-9s %s\n" "$name${relabel:+ relabel}" $c $k $expect "$line" | tee -a $OUT/logs/matrix.txt
      done
    done
  done
done
exit $fail
