#!/usr/bin/env python3
"""Z16: build offline (AOT) Metal binary archives with `metal-tt` for the runtime loader in
risc0/zkp/src/hal/metal/archive.rs (`RISC0_METAL_AOT_DIR`).

usage: build_archive.py --metallib FILE --unit UNIT --names N1,N2,... --arch ARCH [--arch ARCH ...]
                        --out DIR [--metal-tt PATH] [--macos-min 27.0]

Writes DIR/<unit>-<chunked_digest(metallib)[:24]>-<arch>-<checksum>.binarchive per arch, the name
the runtime looks for (`archive::aot_stem` + `archive_file`): the runtime only uses the file of its
own GPU (`MTLDevice.architecture.name`, the same string as `metal-tt -arch`, e.g. applegpu_g17s on
an M5 Max) and of the exact metallib it embeds, and checks the checksum before Metal reads it.

The names must be the kernel set the runtime builds for that unit (e.g. rv32im-sys
`METAL_KERNEL_NAMES`); a kernel not in the archive is simply compiled at runtime as before.
Cost (Z16 note): ~1-2 min wall and ~2.2 GB RAM per arch for rv32im on an M5 Max, ~35 MB per file.
`metal-tt` is in the Metal toolchain (`$METAL_TC/usr/bin/metal-tt`, see env.sh); no GPU is used.
"""
import argparse, hashlib, json, os, shutil, struct, subprocess, sys, tempfile

CHUNK = 1 << 20


def chunked_digest(data: bytes) -> str:
    """archive.rs `parallel_digest`: sha256(le64(len) || sha256(chunk_i)...) as hex."""
    h = hashlib.sha256(struct.pack('<Q', len(data)))
    for i in range(0, len(data), CHUNK):
        h.update(hashlib.sha256(data[i:i + CHUNK]).digest())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--metallib', required=True)
    ap.add_argument('--unit', required=True)
    ap.add_argument('--names', required=True, help='comma-separated kernel names')
    ap.add_argument('--arch', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--metal-tt', default=None)
    ap.add_argument('--macos-min', default='27.0')
    a = ap.parse_args()
    tt = a.metal_tt or shutil.which('metal-tt') or os.path.join(os.environ.get('METAL_TC', ''), 'usr/bin/metal-tt')
    if not os.path.exists(tt):
        sys.exit(f'metal-tt not found ({tt}); pass --metal-tt or set METAL_TC')
    lib = open(a.metallib, 'rb').read()
    digest = chunked_digest(lib)[:24]
    names = [n for n in a.names.split(',') if n]
    os.makedirs(a.out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        libpath = os.path.join(tmp, 'lib.metallib')
        shutil.copyfile(a.metallib, libpath)
        script = os.path.join(tmp, 'pipelines.mtlp-json')
        with open(script, 'w') as f:
            json.dump({'libraries': {'paths': [{'label': 'lib', 'path': libpath}]},
                       'pipelines': {'compute_pipelines': [{'compute_function': f'alias:lib#{n}'} for n in names]}}, f)
        for arch in a.arch:
            raw = os.path.join(tmp, f'{arch}.binarchive')
            subprocess.run([tt, '-arch', arch, '-macos-version-min', a.macos_min, '-o', raw, script], check=True)
            content = open(raw, 'rb').read()
            name = f'{a.unit}-{digest}-{arch}-{chunked_digest(content)[:16]}.binarchive'
            stem = f'{a.unit}-{digest}-{arch}-'
            for old in os.listdir(a.out):
                if old.startswith(stem) and old.endswith('.binarchive'):
                    os.remove(os.path.join(a.out, old))
            dst = os.path.join(a.out, name)
            shutil.copyfile(raw, dst + '.tmp')
            os.replace(dst + '.tmp', dst)
            print(f'{dst} ({len(content)} bytes, {len(names)} kernels)')


if __name__ == '__main__':
    main()
