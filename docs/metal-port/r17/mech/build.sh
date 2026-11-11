#!/bin/sh
# Builds the host. Needs only the Command Line Tools (clang); the shader is compiled at run time by Metal.
set -e
cd "$(dirname "$0")"
clang -fobjc-arc -O2 -arch arm64 -mmacosx-version-min=12.0 -Wno-deprecated-declarations \
  -framework Metal -framework Foundation main.m -o repro
echo "built ./repro"
