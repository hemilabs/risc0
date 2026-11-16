#!/usr/bin/env python3
"""Z16 / W7-Z1: build offline (AOT) Metal binary archives with `metal-tt` for the runtime loader in
risc0/zkp/src/hal/metal/archive.rs (`RISC0_METAL_AOT_DIR`).

usage: build_archive.py --metallib FILE --unit UNIT --names N1,N2,... --arch ARCH [--arch ARCH ...]
                        --out DIR [--fc INDEX=VALUE[,INDEX=VALUE...]] [--os-key major|build|KEY]
                        [--macos-min 27.0] [--metal-tt PATH] [--jobs N] [--manifest FILE]

Writes DIR/<unit>-<digest(metallib)[:24]>-<arch>-<os key>-<checksum>.binarchive per arch, the name the
runtime looks for (`archive::aot_stem` + `aot_file_name`). The runtime only opens the file of its own
GPU (`MTLDevice.architecture.name`, the same string as `metal-tt -arch`, e.g. applegpu_g17s on an M5
Max), of the exact metallib it embeds and of its OS key, and checks the checksum before Metal reads it.
`digest` and `checksum` are the parallel chunked sha256 of archive.rs (`parallel_digest`), in every
`RISC0_METAL_ARCHIVE_HASH` mode.

OS key (`--os-key`):
  major (default)  `macos<N>` with N the major of --macos-min: the packaging form, used on every build
                   of that major version (the runtime tries its exact build first, then this).
  build            `os<ProductBuildVersion>` of this Mac (sw_vers -buildVersion): only this OS build.
  KEY              taken as given (must be `os<build>` or `macos<N>` to ever be used).

--fc builds each name specialized with the given uint function constants (metal-tt
`specialized_functions`), the archive of one `archive::specialized_pipeline` unit (e.g. rv32im's
`rv32im_fc1048576` for `ef_eval_split_fc` with `--fc 0=1048576`), from a library of just those
functions (metallib_extract.py: metal-tt 32023.922 cannot specialize a function whose metadata is
not the library's first). Without --fc the names are built plain. The names must be the kernel set the runtime builds for that unit (rv32im: what
`Rv32imKernels::get` loads); a kernel not in the archive is simply compiled at runtime as before.

Cost (Z16/W7-Z1 notes): ~1-2 min wall and ~2.2 GB RAM per arch for rv32im on an M5 Max, ~35 MB per
file. `metal-tt` is in the Metal toolchain (`$METAL_TC/usr/bin/metal-tt`, see env.sh); no GPU is used.
--manifest appends one JSON line per file written (name, sha256, sizes, inputs, toolchain version).
"""
import argparse, concurrent.futures, hashlib, json, os, shutil, struct, subprocess, sys, tempfile, time

CHUNK = 1 << 20


def chunked_digest(data: bytes) -> str:
    """archive.rs `parallel_digest`: sha256(le64(len) || sha256(chunk_i)...) as hex."""
    h = hashlib.sha256(struct.pack('<Q', len(data)))
    for i in range(0, len(data), CHUNK):
        h.update(hashlib.sha256(data[i:i + CHUNK]).digest())
    return h.hexdigest()


def os_key(arg: str, macos_min: str) -> str:
    if arg == 'major':
        return 'macos' + macos_min.split('.')[0]
    if arg == 'build':
        b = subprocess.run(['sw_vers', '-buildVersion'], capture_output=True, text=True, check=True).stdout.strip()
        return 'os' + b
    return arg


def tt_version(tt: str) -> str:
    r = subprocess.run([tt, '-version'], capture_output=True, text=True)
    return (r.stdout or r.stderr).strip().splitlines()[0] if (r.stdout or r.stderr) else 'unknown'


def parse_fc(s):
    out = []
    for kv in s.split(','):
        k, v = kv.split('=')
        out.append((int(k), int(v)))
    return out


def script_for(libpath, names, fc):
    libs = {'paths': [{'label': 'lib', 'path': libpath}]}
    pipes = []
    if fc:
        libs['specialized_functions'] = []
        for i, n in enumerate(names):
            label = f'spec{i}'
            libs['specialized_functions'].append({
                'label': label,
                'function': f'alias:lib#{n}',
                'constant_values': [{'id_type': 'FunctionConstantIndex', 'id': {'data': k},
                                     'value_type': 'ConstantUInt', 'value': {'data': v}} for k, v in fc],
            })
            pipes.append({'compute_function': f'alias:{label}#{n}'})
    else:
        pipes = [{'compute_function': f'alias:lib#{n}'} for n in names]
    return {'libraries': libs, 'pipelines': {'compute_pipelines': pipes}}


def build_one(tt, arch, a, script, tmp, digest, key, names, fc, version, lib_sha):
    raw = os.path.join(tmp, f'{arch}.binarchive')
    t0 = time.time()
    r = subprocess.run([tt, '-arch', arch, '-macos-version-min', a.macos_min, '-o', raw, script],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return arch, None, f'metal-tt failed for {arch}: {r.stderr.strip()[-2000:]}'
    content = open(raw, 'rb').read()
    stem = f'{a.unit}-{digest}-{arch}-{key}'
    name = f'{stem}-{chunked_digest(content)[:16]}.binarchive'
    for old in os.listdir(a.out):
        if old.startswith(stem + '-') and old.endswith('.binarchive'):
            os.remove(os.path.join(a.out, old))
    dst = os.path.join(a.out, name)
    shutil.copyfile(raw, dst + '.tmp')
    os.replace(dst + '.tmp', dst)
    os.remove(raw)
    rec = {'file': name, 'unit': a.unit, 'arch': arch, 'os_key': key, 'macos_min': a.macos_min,
           'metallib_digest24': digest, 'metallib_sha256': lib_sha, 'names': names,
           'fc': ','.join(f'{k}={v}' for k, v in fc) if fc else None, 'bytes': len(content),
           'sha256': hashlib.sha256(content).hexdigest(), 'metal_tt': version,
           'build_seconds': round(time.time() - t0, 1)}
    return arch, rec, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--metallib', required=True)
    ap.add_argument('--unit', required=True)
    ap.add_argument('--names', required=True, help='comma-separated kernel names')
    ap.add_argument('--arch', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fc', default=None, help='INDEX=VALUE[,...] uint function constants')
    ap.add_argument('--os-key', default='major')
    ap.add_argument('--metal-tt', default=None)
    ap.add_argument('--macos-min', default='27.0')
    ap.add_argument('--jobs', type=int, default=1, help='arches built at once (~2.2 GB RAM each for rv32im)')
    ap.add_argument('--manifest', default=None)
    a = ap.parse_args()
    tt = a.metal_tt or shutil.which('metal-tt') or os.path.join(os.environ.get('METAL_TC', ''), 'usr/bin/metal-tt')
    if not os.path.exists(tt):
        sys.exit(f'metal-tt not found ({tt}); pass --metal-tt or set METAL_TC')
    if '-' in a.unit:
        sys.exit('unit names must not contain "-"')
    lib = open(a.metallib, 'rb').read()
    digest = chunked_digest(lib)[:24]
    lib_sha = hashlib.sha256(lib).hexdigest()
    names = [n for n in a.names.split(',') if n]
    fc = parse_fc(a.fc) if a.fc else []
    key = os_key(a.os_key, a.macos_min)
    if '-' in key:
        sys.exit(f'OS key {key!r} must not contain "-"')
    version = tt_version(tt)
    os.makedirs(a.out, exist_ok=True)
    failed = False
    with tempfile.TemporaryDirectory() as tmp:
        libpath = os.path.join(tmp, 'lib.metallib')
        if fc:
            # metal-tt specializes only a function whose metadata comes first in its library
            # (metallib_extract.py): build from a library of just these functions. The name keeps
            # the digest of the full library, which is what the runtime embeds and looks up.
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import metallib_extract
            open(libpath, 'wb').write(metallib_extract.extract(lib, names))
        else:
            shutil.copyfile(a.metallib, libpath)
        script = os.path.join(tmp, 'pipelines.mtlp-json')
        with open(script, 'w') as f:
            json.dump(script_for(libpath, names, fc), f)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            futs = [ex.submit(build_one, tt, arch, a, script, tmp, digest, key, names, fc, version, lib_sha)
                    for arch in a.arch]
            for fut in concurrent.futures.as_completed(futs):
                arch, rec, err = fut.result()
                if err:
                    print(err, file=sys.stderr)
                    failed = True
                    continue
                print(f"{os.path.join(a.out, rec['file'])} ({rec['bytes']} bytes, {len(names)} kernels, "
                      f"{rec['build_seconds']} s)", flush=True)
                if a.manifest:
                    with open(a.manifest, 'a') as m:
                        m.write(json.dumps(rec) + '\n')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
