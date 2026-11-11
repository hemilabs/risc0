#!/bin/sh
# Runs the reproducer on any Apple Silicon Mac (macOS 12 or later) and saves the output.
# Needs nothing beyond macOS itself when the prebuilt ./repro is used; ./build.sh rebuilds it with the
# Command Line Tools (xcode-select --install). The shader is compiled at run time by the installed Metal.
# Takes about 1-3 minutes. Close other GPU-heavy apps for a cleaner rate (the bug shows up either way on M5 Max).
cd "$(dirname "$0")"
xattr -d com.apple.quarantine ./repro 2>/dev/null
if [ ! -x ./repro ] || [ "$(uname -m)" != arm64 ]; then ./build.sh || exit 1; fi
model=$(sysctl -n hw.model 2>/dev/null || echo unknown)
out="result-$model-$(date +%Y%m%d-%H%M%S).txt"
{
  echo "== default table (1048576 threads x 10 dispatches, threadgroup 256)"; ./repro
  echo; echo "== dispatch-size sweep"; ./repro --sweep
  echo; echo "== stress (output layouts, 1M/4M threads, threadgroup 256/1024)"; ./repro --stress
} 2>&1 | tee "$out"
echo; echo "saved $out"
