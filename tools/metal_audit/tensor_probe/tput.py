#!/usr/bin/env python3
# N1: run the tput.metal throughput sweep with host.m and print rates.
#   tput.py <host> <tput.metallib> <rounds> [filter]
# Reports MAC/s for the mm_* kernels and op/s for the s_* kernels, and Baby Bear field products/s:
# a field product = 16 byte MACs on the matrix path (the RDNA4 byte decomposition), and on the
# scalar path one s_link (the chain link's 64-bit multiply-add + fold) or one s_mont.
import re, subprocess, sys
host, lib, rounds = sys.argv[1], sys.argv[2], sys.argv[3]
flt = sys.argv[4] if len(sys.argv) > 4 else ''
NTG = 4096
specs, work = [], {}
shapes = [(16,16,16,1),(16,32,64,1),(32,32,32,1),(32,32,32,4),(64,32,32,4),(64,64,64,4)]
for src, tn in [('tg','u8'),('tg','i8'),('tg','h'),('dev','u8'),('dev','h')]:
    extra = [(16,32,128,1)] if (src, tn) == ('tg','u8') else [(16,32,256,1)] if (src, tn) == ('dev','u8') else []
    for M,N,K,SG in shapes + extra:
        n = f'mm_{src}_{tn}_{M}x{N}x{K}_{SG}'
        if flt and not re.search(flt, n): continue
        it = max(16, int(2**31 // (M*N*K*NTG*(4 if SG == 1 else 1)) * 4))
        specs.append(f'{n}:128:{NTG}:{it}:{NTG*4*M*N*4}:65536')
        work[n] = ('MAC', M*N*K*it*NTG*(4 if SG == 1 else 1))
for s in ['s_mul32','s_mulhi','s_wmac','s_link','s_mont','s_fma']:
    if flt and not re.search(flt, s): continue
    it = 4096
    specs.append(f'{s}:256:{NTG}:{it}:{NTG*256*4}:65536')
    work[s] = ('op', 8*it*NTG*256)
out = subprocess.run([host, lib, rounds] + specs, capture_output=True, text=True)
sys.stdout.write(out.stderr)
rates = {}
for L in out.stdout.splitlines():
    m = re.match(r'TIME (\S+) .* med=([\d.]+) min=([\d.]+) max=([\d.]+)', L)
    if not m:
        print(L); continue
    n, med, mn, mx = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
    kind, w = work[n]
    r = w / (mn * 1e-3)
    rates[n] = r
    print(f'{n:28s} med {med:8.3f} ms [{mn:.3f}, {mx:.3f}]  {r/1e12:7.2f} T{kind}/s (at min)')
link = rates.get('s_link'); mont = rates.get('s_mont')
if link:
    print(f'\nscalar: chain link (64-bit madd+fold) {link/1e12:.2f} T/s = field products/s; montmul {mont/1e12 if mont else 0:.2f} T/s')
    for n, r in rates.items():
        if n.startswith('mm_') and '_h_' not in n:
            fp = r / 16
            print(f'  {n:28s} {fp/1e12:6.3f} T field products/s = {fp/link:5.2f}x the scalar link (before staging)')
