# Remove the hoisted _h_data_* scalar parameters from rv32im_v2_* signatures and calls
# (valid only after `sched.py unhoist`, which leaves them unused in the callees).
import re, sys
src = open(sys.argv[1]).read()
cut = src.find('// Intel SYCL eval_check kernel')
head, tail = src[:cut], src[cut:]
out = []
for l in head.split('\n'):
    if re.match(r'^(__attribute__\(\(noinline\)\) )?FpExt rv32im_v2_\d+\(', l):
        l = re.sub(r', Fp _h_data_\d+_\d+', '', l)
    elif re.search(r'\brv32im_v2_\d+\(cycle', l):
        l = re.sub(r', _h_data_\d+_\d+', '', l)
    out.append(l)
h = '\n'.join(out)
# sanity: no _h_data use left inside sub-functions
open(sys.argv[2], 'w').write(h + tail)
