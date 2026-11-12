#!/usr/bin/env python3
# idx32.py in.cpp out.cpp  (final post-slab amalgamation)
# Column loads `buf[C * steps + ((cycle - kInvRate * B) & mask)]` use 64-bit math on
# `steps`, which is a stack-call argument and therefore per-lane in IGC: every load paid an
# emulated 64-bit multiply (GenISA.mul.pair) plus 64-bit adds. Rewrite them with 32-bit
# unsigned index math (C * steps < 211 * 2^24 < 2^32 for domain <= 2^24, i.e. po2 <= 22;
# the kernel wrapper asserts the bound).
import re, sys
src = open(sys.argv[1]).read()
k = '// Intel SYCL eval_check kernel'
head, tail = src[:src.find(k)], src[src.find(k):]
n0 = head.count('size_t mask = steps - 1;')
head = head.replace('size_t mask = steps - 1;', 'size_t mask = steps - 1; const uint32_t rk_s = (uint32_t)steps, rk_m = rk_s - 1u, rk_c = (uint32_t)cycle;')
pat = re.compile(r'\[(\d+) \* steps \+ \(\(cycle - kInvRate \* (\d+)\) & mask\)\]')
max_c = 0
def _rw(m):
    global max_c
    max_c = max(max_c, int(m.group(1)))
    return '[(uint32_t)(%su * rk_s + ((rk_c - %du) & rk_m))]' % (m.group(1), 4 * int(m.group(2)))
head, n = pat.subn(_rw, head)
assert 'steps + ((cycle' not in head
# 32-bit index safety. The largest index is max_c * steps + (steps - 1), and the
# kernel-wrapper guard below enforces steps (= domain) <= 2^24, so the worst case is
# (max_c + 1) * 2^24 - 1. That must stay < 2^32, i.e. max_c < 256. Assert it at codegen
# so a future circuit regen that introduces a column >= 256 fails loudly here instead of
# silently wrapping the uint32 index at large po2 (which would still pass small-po2
# bit-exact tests). The pin above counts loads, not column magnitude, so it can't catch this.
assert max_c < 256, (
    'idx32: max trace column %d >= 256 overflows the 32-bit index at domain=2^24; '
    'widen this rewrite back to 64-bit before shipping' % max_c)
guard = 'uint32_t global_size = ((rows + WG_SIZE - 1) / WG_SIZE) * WG_SIZE;'
assert tail.count(guard) == 1
tail = tail.replace(guard, guard + '\n        if (domain > (1u << 24)) throw std::runtime_error("eval_check: idx32 kernel supports domain <= 2^24");')
open(sys.argv[2], 'w').write(head + tail)
print('idx32: %d functions, %d loads' % (n0, n), file=sys.stderr)
