#!/usr/bin/env python3
# N1: the table buffer for chain.metal: pm rows (raw Montgomery words < P) and the Q tiles of the
# byte decomposition for chains of L links in blocks of BL links.   tables.py <out> <L> <BL> <NCH>
import random, struct, sys
P = 0x78000001; NK = 1255
out, L, BL, NCH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rnd = random.Random(1)
pm = [[rnd.randrange(P) for _ in range(4)] for _ in range(NK)]
buf = bytearray()
for r in pm: buf += struct.pack('<4I', *r)
K = 4 * BL; NB = L // BL
for ch in range(NCH):
    for bi in range(NB):
        tile = bytearray(16 * K)
        for l in range(BL):
            k = ((ch * L + bi * BL + l) * 7) % NK
            for j in range(4):
                for a in range(4):
                    v = pm[k][j] * pow(256, a, P) % P
                    for b in range(4):
                        tile[(4 * j + b) * K + 4 * l + a] = (v >> (8 * b)) & 255
        buf += tile
open(out, 'wb').write(buf)
