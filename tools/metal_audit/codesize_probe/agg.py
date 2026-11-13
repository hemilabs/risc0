#!/usr/bin/env python3
"""agg.py LOG...: pool the 'R21CS raw round' lines of bench runs (optionally only sessions that
started on an idle GPU: lines after a 'R21CS session idle' marker) and print, per kernel, the
min / median time and the median per-round ratio to the first kernel of the round."""
import re, statistics, sys
idle_only = "--idle-only" in sys.argv
rows, ok = [], True
for f in [a for a in sys.argv[1:] if not a.startswith("--")]:
    for line in open(f, errors="replace"):
        if line.startswith("R21CS session"):
            ok = (not idle_only) or (" idle" in line and "NOT" not in line)
        m = re.match(r"R21CS raw round \d+ (.*)", line)
        if m and ok:
            rows.append([(int(k), float(v)) for k, v in (x.split(":") for x in m.group(1).split())])
if not rows:
    sys.exit("no rounds")
n = len(rows[0])
print(f"rounds {len(rows)}")
for i in range(n):
    b = rows[0][i][0]
    ts = sorted(r[i][1] for r in rows)
    rs = sorted(r[i][1] / r[0][1] for r in rows)
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
    print(f"B={b:6d} col{i}  min {ts[0]:8.3f}  p10 {q(ts,.1):8.3f}  med {statistics.median(ts):8.3f}  "
          f"ratio med {statistics.median(rs):.3f}  p25-p75 [{q(rs,.25):.3f}, {q(rs,.75):.3f}]")
