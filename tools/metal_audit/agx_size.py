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
  agx_size.py ... --fc 0=4194304 LIB FN     (R22: FN specialized with uint function constants)
  agx_size.py --parse OBJ.agx ...           (metadata of objects kept with --keep)
  agx_size.py --archive FILE.binarchive ... (R22: the pipelines of runtime MTLBinaryArchive
                                             files, i.e. what the OS driver compiled)
  agx_size.py ... --guard-stack BYTES ...   (R22: exit 2 if a function's stack exceeds BYTES)
Output: one line per function, tab separated: lib, function, text_bytes, metadata, seconds.
Metadata (see META_FIELDS; empirically identified, treat as tentative): stack = per-thread private
stack bytes including callees, frame = the kernel's own frame, regs? / uregs? = unconfirmed.

R22 regime guard: R21 found that ef_eval_split's time follows this stack field in two clusters
(fast 18-21 KB, ~160 ms at po2 18; slow 32-40 KB, ~400 ms). `--guard-stack 24576` fails when a
toolchain / generator change moves the kernel into the slow cluster. rv32im-sys/build.rs runs it
on every metallib build (RISC0_METAL_EF_STACK_GUARD); the rv32im runtime reads the same field
from the binary archive the OS driver wrote (eval_regime.rs).
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


FAT_MAGICS = (0xCAFEBABE, 0xCBFEBABE)  # 0xCBFEBABE: MTLBinaryArchive files (big-endian fat header)
CPU_APPLEGPU = 0x01000013  # the native slice of an archive (0x01000017: the AIR metallib slice)


def fat_slices(blob):
    """[(cputype, bytes)] of a fat image (a plain image is one slice with cputype None)."""
    magic = struct.unpack_from(">I", blob, 0)[0]
    if magic not in FAT_MAGICS:
        return [(None, blob)]
    n = struct.unpack_from(">I", blob, 4)[0]
    out = []
    for i in range(n):
        cpu, _sub, off, size, _align = struct.unpack_from(">iiIII", blob, 8 + 20 * i)
        out.append((cpu, blob[off:off + size]))
    return out


def ld_names(inner):
    """The length-prefixed strings of a native object's __GPU_LD_MD section (the pipeline's
    function name and its stage, e.g. ["ef_eval_split", "compute"])."""
    try:
        ld = [s for s in sections(inner) if s[0] == "__GPU_LD_MD"]
    except Exception:  # noqa: BLE001
        return []
    if not ld:
        return []
    d = inner[ld[0][2]:ld[0][2] + ld[0][3]]
    out = []
    for o in range(0, len(d) - 4, 4):
        n = struct.unpack_from("<I", d, o)[0]
        if 0 < n < 256 and o + 4 + n < len(d) and d[o + 4 + n] == 0:
            w = d[o + 4:o + 4 + n]
            if all(0x21 <= c < 0x7F for c in w):
                out.append(w.decode())
    return out


def archive_objects(blob):
    """[(function name or None, native object bytes)] of every pipeline in a binary archive (or
    a metal-tt output): the nested applegpu objects of the native slice's __compute section."""
    out = []
    for cpu, sl in fat_slices(blob):
        if cpu not in (None, CPU_APPLEGPU):
            continue
        try:
            comp = [s for s in sections(sl) if s[1] == "__compute"]
        except ValueError:
            continue
        for _seg, _sect, off, size in comp:
            d = sl[off:off + size]
            starts = []
            o = d.find(struct.pack("<I", MH_MAGIC_64))
            while o >= 0:
                starts.append(o)
                o = d.find(struct.pack("<I", MH_MAGIC_64), o + 4)
            # Keep the starts that parse as an object (a magic inside code is not one).
            objs = []
            for i, st in enumerate(starts):
                end = starts[i + 1] if i + 1 < len(starts) else len(d)
                try:
                    sections(d[st:end])
                    objs.append(st)
                except Exception:  # noqa: BLE001
                    pass
            for i, st in enumerate(objs):
                end = objs[i + 1] if i + 1 < len(objs) else len(d)
                inner = d[st:end]
                names = [n for n in ld_names(inner) if n not in ("compute", "vertex", "fragment")]
                out.append((names[0] if names else None, inner))
    return out


def text_bytes(inner):
    return sum(s[3] for s in sections(inner) if s[1] == "__text")


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


def pipeline_script(lib, fn, fc=()):
    """The metal-tt pipeline script of one compute pipeline of `fn` (specialized with the uint
    function constants `fc`, [(index, value)], if any)."""
    libs = {"paths": [{"label": "lib", "path": os.path.abspath(lib)}]}
    ref = f"alias:lib#{fn}"
    if fc:
        libs["specialized_functions"] = [{
            "label": "spec",
            "function": ref,
            "specialized_name": fn,
            "constant_values": [{
                "id_type": "FunctionConstantIndex",
                "id": {"data": i},
                "value_type": "ConstantUInt",
                "value": {"data": v},
            } for i, v in fc],
        }]
        ref = f"alias:spec#{fn}"
    return {"libraries": libs, "pipelines": {"compute_pipelines": [{"compute_function": ref}]}}


def size_one(tt, lib, fn, arch, macos, keep, timeout=None, fc=()):
    work = tempfile.mkdtemp(prefix="agxsz-")
    try:
        script = pipeline_script(lib, fn, fc)
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
            return (lib, fn, None, [f"timeout after {timeout} s"], time.time() - t0, {})
        dt = time.time() - t0
        if r.returncode != 0 or not os.path.exists(outp):
            return (lib, fn, None, (r.stderr or r.stdout).strip().splitlines()[-1:], dt, {})
        blob = open(outp, "rb").read()
        comp = [s for s in sections(blob) if s[1] == "__compute"]
        if len(comp) != 1:
            return (lib, fn, None, [f"expected one __compute section, got {len(comp)}"], dt, {})
        _, _, off, size = comp[0]
        inner = blob[off:off + size]
        secs = sections(inner)
        text = sum(s[3] for s in secs if s[1] == "__text")
        other = [f"{k}={v}" for k, v in native_meta(inner).items()]
        if keep:
            os.makedirs(keep, exist_ok=True)
            tag = "".join(f".fc{i}={v}" for i, v in fc)
            base = os.path.join(keep, f"{os.path.basename(lib)}.{fn}{tag}")
            shutil.copy(outp, base + ".archive")
            open(base + ".agx", "wb").write(inner)
        return (lib, fn, text, other, dt, native_meta(inner))
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


def parse_fn(spec, fc):
    """`FN` or `FN@I=V[,I=V...]` -> (FN, [(I, V)]); the global --fc applies to plain names."""
    if "@" not in spec:
        return spec, list(fc)
    fn, rest = spec.split("@", 1)
    return fn, [tuple(int(x, 0) for x in kv.split("=", 1)) for kv in rest.split(",") if kv]


def guard_verdict(fn, meta, limit):
    """None if `fn` passes the stack guard, else the message (R22). An absent stack field is a
    zero stack (FlatBuffers omits default values, e.g. ef_combine); no metadata at all is
    unknown."""
    if not meta:
        return f"{fn}: native stack size unknown (metadata layout not understood)"
    st = meta.get("stack", 0)
    if st > limit:
        return (f"{fn}: native per-thread stack {st} B > {limit} B: the slow compiler regime "
                f"(R21: ~2.4x eval_check time); see docs/metal-port/notes/R22.md")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="applegpu_g17s")
    ap.add_argument("--macos", default="27.0")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--keep", default=None, help="keep the archives and native objects here")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--timeout", type=float, default=None, help="seconds per function")
    ap.add_argument("--fc", action="append", default=[], metavar="I=V",
                    help="uint function constant for every plain FN (FN@I=V sets it per FN)")
    ap.add_argument("--guard-stack", type=int, default=None, metavar="BYTES",
                    help="exit 2 if a function's native stack exceeds BYTES (3: unknown)")
    ap.add_argument("--only", action="append", default=[], help="--archive: only these names")
    ap.add_argument("lib")
    ap.add_argument("fns", nargs="*")
    ap.add_argument("--parse", action="store_true", help="arguments are kept .agx objects")
    ap.add_argument("--archive", action="store_true",
                    help="arguments are binary archives (runtime MTLBinaryArchive files)")
    a = ap.parse_args()
    fails = []
    if a.parse:
        for p in [a.lib] + a.fns:
            inner = open(p, "rb").read()
            meta = native_meta(inner)
            print(f"{p}\t{text_bytes(inner)}\t{' '.join(f'{k}={v}' for k, v in meta.items())}")
            if a.guard_stack is not None:
                v = guard_verdict(p, meta, a.guard_stack)
                if v:
                    fails.append(v)
    elif a.archive:
        for p in [a.lib] + a.fns:
            for name, inner in archive_objects(open(p, "rb").read()):
                if a.only and name not in a.only:
                    continue
                meta = native_meta(inner)
                print(f"{p}\t{name}\t{text_bytes(inner)}\t"
                      f"{' '.join(f'{k}={v}' for k, v in meta.items())}", flush=True)
                if a.guard_stack is not None:
                    v = guard_verdict(name, meta, a.guard_stack)
                    if v:
                        fails.append(v)
    else:
        fc = [tuple(int(x, 0) for x in kv.split("=", 1)) for kv in a.fc]
        if a.lib.endswith(".air"):
            # One .air linked on its own: metal-tt cannot specialize a function of a metallib
            # linked from several .air modules ("cannot find private metadata"), but can one of
            # a single-module library.
            tmp_air = tempfile.mkdtemp(prefix="agxsz-air-")
            import atexit
            atexit.register(shutil.rmtree, tmp_air, True)
            linked = os.path.join(tmp_air, "lib.metallib")
            r = subprocess.run([toolchain_bin("metallib"), a.lib, "-o", linked],
                               capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"agx_size: metallib {a.lib} failed: {r.stderr.strip()}")
            a.lib = linked
        fns = [parse_fn(f, fc) for f in (kernels_of(a.lib) if a.all else a.fns)]
        if not fns:
            sys.exit("agx_size: no functions given")
        tt = toolchain_bin("metal-tt")
        rc = 0
        with cf.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            futs = [ex.submit(size_one, tt, a.lib, fn, a.arch, a.macos, a.keep, a.timeout, spec)
                    for fn, spec in fns]
            for f in cf.as_completed(futs):
                lib, fn, text, other, dt, meta = f.result()
                if text is None:
                    rc = 1
                print(f"{lib}\t{fn}\t{text if text is not None else 'ERR'}\t{' '.join(other)}"
                      f"\t{dt:.1f}s", flush=True)
                if a.guard_stack is not None:
                    v = guard_verdict(fn, meta, a.guard_stack) if text is not None else \
                        f"{fn}: not translated ({' '.join(other)})"
                    if v:
                        fails.append(v)
        if rc and not fails:
            sys.exit(rc)
    for v in fails:
        print(f"agx_size: REGIME GUARD: {v}", file=sys.stderr, flush=True)
    if fails:
        sys.exit(3 if all("unknown" in v or "not translated" in v for v in fails) else 2)


if __name__ == "__main__":
    main()
