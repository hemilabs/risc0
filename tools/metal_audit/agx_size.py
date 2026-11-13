#!/usr/bin/env python3
"""Native (AGX) code size of Metal compute functions, without Xcode.app (R21 probe tooling).

For each requested function of a .metallib, write a one-pipeline Metal pipeline script
(.mtlp-json), translate it offline with the toolchain's `metal-tt -arch <gpu arch>` into a
binary archive, take the archive's `__compute` section (a nested applegpu Mach-O object) and
report the sizes of that object's sections. `__text` is the native machine code of the
pipeline (the kernel plus every non-inlined callee).

Not on any production path. The offline translator is the same back end the driver runs, but
it is a separate copy of it, so treat the sizes as the toolchain's, not the OS driver's.

Usage:
  agx_size.py [--arch applegpu_g17s] [--macos 27.0] [--jobs N] [--keep DIR] LIB.metallib FN [FN...]
  agx_size.py ... LIB.metallib --all       (every kernel listed by `metal-nm` / the lib's names)
  agx_size.py --parse OBJ.agx ...           (metadata of objects kept with --keep)
Output: one line per function, tab separated: lib, function, text_bytes, metadata, seconds.
Metadata (see META_FIELDS; empirically identified, treat as tentative): stack = per-thread private
stack bytes including callees, frame = the kernel's own frame, regs? / uregs? = unconfirmed.
"""
import argparse, concurrent.futures as cf, json, os, shutil, struct, subprocess, sys, tempfile, time

LC_SEGMENT_64 = 0x19
MH_MAGIC_64 = 0xFEEDFACF


def toolchain_bin(name):
    p = shutil.which(name)
    if p:
        return p
    out = subprocess.run(["xcrun", "-f", name], capture_output=True, text=True)
    if out.returncode == 0 and out.stdout.strip():
        return out.stdout.strip()
    sys.exit(f"agx_size: cannot find {name} (source env.sh so the Metal toolchain is on PATH)")


def sections(blob):
    """[(segname, sectname, offset, size)] of a 64-bit Mach-O image held in `blob`."""
    magic, _cpu, _sub, _ftype, ncmds, _sz, _flags, _res = struct.unpack_from("<IiiIIIII", blob, 0)
    if magic != MH_MAGIC_64:
        raise ValueError("not a 64-bit Mach-O")
    out, off = [], 32
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from("<II", blob, off)
        if cmd == LC_SEGMENT_64:
            nsects = struct.unpack_from("<I", blob, off + 64)[0]
            s = off + 72
            for _ in range(nsects):
                sect = blob[s:s + 16].split(b"\0")[0].decode()
                seg = blob[s + 16:s + 32].split(b"\0")[0].decode()
                _addr, size, foff = struct.unpack_from("<QQI", blob, s + 32)
                out.append((seg, sect, foff, size))
                s += 80
        off += cmdsize
    return out


def _fb_table_fields(d, t):
    """{field index: (byte offset, u32)} of the FlatBuffers table at byte t of d (no schema)."""
    vt = t - struct.unpack_from("<i", d, t)[0]
    vsz = struct.unpack_from("<H", d, vt)[0]
    out = {}
    for i in range((vsz - 4) // 2):
        fo = struct.unpack_from("<H", d, vt + 4 + 2 * i)[0]
        if fo and t + fo + 4 <= len(d):
            out[i] = (t + fo, struct.unpack_from("<I", d, t + fo)[0])
    return out


# Fields of the native object's first __compute section (a FlatBuffers blob), root table -> field 0
# -> table. Identified empirically, NOT from a schema (R21):
#  14: per-thread private stack bytes including the deepest call chain. Calibrated with kernels
#      holding a dynamically indexed private uint[N] (value 4N + 16), and o1/os eval variants whose
#      PSO creation fails with "exceeds available stack space" report 320-412 KB.
#  41: the kernel's own frame (equals 14 when nothing is called out of line).
#   1: 4..256; 256 for ef_eval_split. Tentatively the register allocation (16-bit units?).
#   3: tentatively uniform registers.
META_FIELDS = {"stack": 14, "frame": 41, "regs?": 1, "uregs?": 3}


def native_meta(inner):
    """Selected metadata of a native applegpu object (see META_FIELDS); {} if not parseable."""
    try:
        comp = [s for s in sections(inner) if s[1] == "__compute"]
        d = inner[comp[0][2]:comp[0][2] + comp[0][3]]
        root = struct.unpack_from("<I", d, 0)[0]
        rf = _fb_table_fields(d, root)
        o, v = rf[0]
        f = _fb_table_fields(d, o + v)
        return {k: f[i][1] for k, i in META_FIELDS.items() if i in f}
    except Exception:  # noqa: BLE001 - layout not understood: report nothing
        return {}


def size_one(tt, lib, fn, arch, macos, keep, timeout=None):
    work = tempfile.mkdtemp(prefix="agxsz-")
    try:
        script = {
            "libraries": {"paths": [{"label": "lib", "path": os.path.abspath(lib)}]},
            "pipelines": {"compute_pipelines": [{"compute_function": f"alias:lib#{fn}"}]},
        }
        js = os.path.join(work, "p.mtlp-json")
        with open(js, "w") as f:
            json.dump(script, f)
        outp = os.path.join(work, "out.bin")
        t0 = time.time()
        try:
            # start_new_session + killpg: metal-tt spawns air-tt / applegpu-nt children.
            pr = subprocess.Popen([tt, "-arch", arch, "-macos-version-min", macos, js, "-o", outp],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  start_new_session=True)
            so, se = pr.communicate(timeout=timeout)
            r = subprocess.CompletedProcess(pr.args, pr.returncode, so, se)
        except subprocess.TimeoutExpired:
            os.killpg(pr.pid, 9)
            pr.communicate()
            return (lib, fn, None, [f"timeout after {timeout} s"], time.time() - t0)
        dt = time.time() - t0
        if r.returncode != 0 or not os.path.exists(outp):
            return (lib, fn, None, (r.stderr or r.stdout).strip().splitlines()[-1:], dt)
        blob = open(outp, "rb").read()
        comp = [s for s in sections(blob) if s[1] == "__compute"]
        if len(comp) != 1:
            return (lib, fn, None, [f"expected one __compute section, got {len(comp)}"], dt)
        _, _, off, size = comp[0]
        inner = blob[off:off + size]
        secs = sections(inner)
        text = sum(s[3] for s in secs if s[1] == "__text")
        other = [f"{k}={v}" for k, v in native_meta(inner).items()]
        if keep:
            os.makedirs(keep, exist_ok=True)
            base = os.path.join(keep, f"{os.path.basename(lib)}.{fn}")
            shutil.copy(outp, base + ".archive")
            open(base + ".agx", "wb").write(inner)
        return (lib, fn, text, other, dt)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def kernels_of(lib):
    nm = toolchain_bin("metal-nm")
    r = subprocess.run([nm, lib], capture_output=True, text=True)
    names = []
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-2] in ("T", "t") and not parts[-1].startswith("_"):
            names.append(parts[-1])
    return sorted(set(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="applegpu_g17s")
    ap.add_argument("--macos", default="27.0")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--keep", default=None, help="keep the archives and native objects here")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--timeout", type=float, default=None, help="seconds per function")
    ap.add_argument("lib")
    ap.add_argument("fns", nargs="*")
    ap.add_argument("--parse", action="store_true", help="arguments are kept .agx objects")
    a = ap.parse_args()
    if a.parse:
        for p in [a.lib] + a.fns:
            inner = open(p, "rb").read()
            text = sum(s[3] for s in sections(inner) if s[1] == "__text")
            meta = " ".join(f"{k}={v}" for k, v in native_meta(inner).items())
            print(f"{p}\t{text}\t{meta}")
        return
    fns = kernels_of(a.lib) if a.all else a.fns
    if not fns:
        sys.exit("agx_size: no functions given")
    tt = toolchain_bin("metal-tt")
    rc = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = [ex.submit(size_one, tt, a.lib, fn, a.arch, a.macos, a.keep, a.timeout)
                for fn in fns]
        for f in cf.as_completed(futs):
            lib, fn, text, other, dt = f.result()
            if text is None:
                rc = 1
            print(f"{lib}\t{fn}\t{text if text is not None else 'ERR'}\t{' '.join(other)}\t{dt:.1f}s",
                  flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
