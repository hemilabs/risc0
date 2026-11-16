#!/usr/bin/env bash
# N1: build the probe host and metallibs into <out dir> (source ~/zkminer/env.sh first).
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); O=${1:?out dir}; mkdir -p "$O"
clang -O2 -fobjc-arc -framework Foundation -framework Metal "$D/host.m" -o "$O/host"
mk() { local src=$1 name=$2; shift 2
  xcrun metal -std=metal4.0 -O2 "$@" -c "$D/$src" -o "$O/$name.air"; xcrun metallib "$O/$name.air" -o "$O/$name.metallib"; }
mk tput.metal tput
mk chain.metal chain
mk chain.metal chain_noload -DCH_NOLOAD
mk layout.metal layout
