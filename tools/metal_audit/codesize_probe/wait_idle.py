#!/usr/bin/env python3
"""wait_idle.py [max_s] [threshold] [samples]: wait until the GPU 'Device Utilization %' (ioreg,
AGXAccelerator) stays below `threshold` (15) for `samples` (4) consecutive 0.5 s samples.
Exit 0 when idle, 1 on timeout. Prints the outcome. (Copy of R19's scratch tool.)"""
import re, subprocess, sys, time
mx = float(sys.argv[1]) if len(sys.argv) > 1 else 300
th = int(sys.argv[2]) if len(sys.argv) > 2 else 15
need = int(sys.argv[3]) if len(sys.argv) > 3 else 4


def util():
    o = subprocess.run(["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"], capture_output=True, text=True).stdout
    m = re.search(r'"Device Utilization %"=(\d+)', o)
    return int(m.group(1)) if m else -1


t0, ok = time.time(), 0
while time.time() - t0 < mx:
    u = util()
    ok = ok + 1 if 0 <= u < th else 0
    if ok >= need:
        print("idle after %.0f s" % (time.time() - t0))
        sys.exit(0)
    time.sleep(0.5)
print("NOT idle after %.0f s (util %d)" % (mx, util()))
sys.exit(1)
