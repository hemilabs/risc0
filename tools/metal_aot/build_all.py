#!/usr/bin/env python3
"""W7-Z1: build the full offline (AOT) archive set of one risc0 build for several GPU architectures.

usage: build_all.py --dump DIR [--recursion-metallib FILE --recursion-sys-lib FILE]
                    --arch ARCH [--arch ARCH ...] --out DIR [--os-key major] [--macos-min 27.0]
                    [--jobs N] [--only UNIT[,UNIT...]] [--recursion-variants rg|all]

Inputs:
  --dump DIR  written by the `w7z1_dump_aot_inputs` tests (zkp, rv32im, keccak, groth16-sys; run each
              test binary with `--ignored --exact <path>::w7z1_dump_aot_inputs` and `W7Z1_DUMP=DIR`):
              `units.jsonl` (one line per unit: unit, metallib file, kernel names, function
              constants) and the embedded metallibs, byte for byte what the binary loads.
  recursion   has no dump test (its HAL module belongs to another item): pass the metallib the build
              embedded (`$OUT_DIR` of risc0-circuit-recursion-sys, `RECURSION_METAL_PATH`) and its
              lib.rs; units `recursion` (every non-eval_check kernel of METAL_KERNEL_NAMES) and
              `recursion_splitrg` (the default eval_check variant; `--recursion-variants all` adds the
              revert variants).

Writes every archive to --out via build_archive.py, appends `manifest.jsonl` there (one JSON line per
file: name, unit, arch, OS key, sizes, sha256, metallib digest, toolchain version) and rewrites
`SHA256SUMS` (shasum -a 256 -c format) over every .binarchive in --out.
"""
import argparse, hashlib, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))

REC_VARIANTS = {
    'recursion_splitrg': lambda n: n.startswith('ec_rg_part_'),
    'recursion_splitlz': lambda n: n.startswith('ec_lz_part_'),
    'recursion_splitlzu32': lambda n: n.startswith('ec_lz32_part_'),
    'recursion_split': lambda n: re.fullmatch(r'ec_part_\d+', n) is not None,
    'recursion_monolith': lambda n: n == 'eval_check',
}


def recursion_units(lib_rs, metallib, variants):
    src = open(lib_rs).read()
    body = src.split('pub const METAL_KERNEL_NAMES: &[&str] = &[', 1)[1].split('];', 1)[0]
    body = '\n'.join(l.split('//', 1)[0] for l in body.splitlines())
    names = re.findall(r'"([A-Za-z0-9_]+)"', body)
    is_eval = lambda n: n == 'eval_check' or n.startswith('ec_')
    units = [{'unit': 'recursion', 'metallib': metallib, 'names': [n for n in names if not is_eval(n)], 'fc': None}]
    for unit, pick in REC_VARIANTS.items():
        if variants == 'all' or unit == 'recursion_splitrg':
            units.append({'unit': unit, 'metallib': metallib, 'names': [n for n in names if pick(n)], 'fc': None})
    return units


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', required=True)
    ap.add_argument('--recursion-metallib')
    ap.add_argument('--recursion-sys-lib')
    ap.add_argument('--recursion-variants', default='rg', choices=['rg', 'all'])
    ap.add_argument('--arch', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--os-key', default='major')
    ap.add_argument('--macos-min', default='27.0')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    units = []
    with open(os.path.join(a.dump, 'units.jsonl')) as f:
        for line in f:
            if line.strip():
                u = json.loads(line)
                u['metallib'] = os.path.join(a.dump, u['metallib'])
                units.append(u)
    if a.recursion_metallib:
        units += recursion_units(a.recursion_sys_lib, a.recursion_metallib, a.recursion_variants)
    seen = {}
    for u in units:  # a dump appended twice: keep the last line of each unit
        seen[u['unit']] = u
    units = list(seen.values())
    if a.only:
        keep = set(a.only.split(','))
        units = [u for u in units if u['unit'] in keep]
    os.makedirs(a.out, exist_ok=True)
    manifest = os.path.join(a.out, 'manifest.jsonl')
    failed = []
    for u in units:
        cmd = [sys.executable, os.path.join(HERE, 'build_archive.py'), '--metallib', u['metallib'],
               '--unit', u['unit'], '--names', ','.join(u['names']), '--out', a.out,
               '--os-key', a.os_key, '--macos-min', a.macos_min, '--jobs', str(a.jobs),
               '--manifest', manifest]
        if u.get('fc'):
            cmd += ['--fc', u['fc']]
        for arch in a.arch:
            cmd += ['--arch', arch]
        print(f"== {u['unit']}: {len(u['names'])} kernels x {len(a.arch)} arches", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(u['unit'])
    sums = []
    for name in sorted(os.listdir(a.out)):
        if name.endswith('.binarchive'):
            h = hashlib.sha256(open(os.path.join(a.out, name), 'rb').read()).hexdigest()
            sums.append(f'{h}  {name}\n')
    with open(os.path.join(a.out, 'SHA256SUMS'), 'w') as f:
        f.writelines(sums)
    if failed:
        sys.exit(f'failed units: {failed}')


if __name__ == '__main__':
    main()
