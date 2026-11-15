#!/usr/bin/env python3
"""W7-Z1: write a metallib holding only some functions of another one (each with its own metadata and
bitcode module, byte for byte), so `metal-tt` can specialize them.

usage: metallib_extract.py IN.metallib OUT.metallib NAME [NAME ...]
       metallib_extract.py --list IN.metallib

Why: metal-tt 32023.922 (Metal toolchain 27B47) fails to specialize a function with function
constants (`specialized_functions`) unless the function's private metadata is the first in the
metallib ("applegpu-nt: error: cannot find private metadata at offset 'N'"; reproduced with a two-kernel
library, W7-Z1 note). The function's bitcode, hash and metadata are unchanged by the extraction, and
the runtime's archive lookup matched an archive built from such a library (W7-Z1 note), so an archive
built from OUT serves PSOs of the original library's function.

Format (MTLB, file version 1.2+ as the 27.x toolchain writes it): a header with (offset, size) u64
pairs of the function list (0x18), the public metadata (0x28), the private metadata (0x38) and the
bitcode (0x48); the function list is a u32 count (not counted by its size field) and per function a u32 size (counting itself) plus tags (4-byte name,
u16 length, data; `ENDT` ends one), among them OFFT = (public, private, bitcode) offsets of the
function's parts in those sections and MDSZ = its bitcode size. A tag list after the function list
(the header extension, `ENDT`-terminated) is copied when it holds nothing but ENDT and refused
otherwise.
"""
import struct, sys


def read_tags(b, pos, end):
    tags = []
    while pos < end:
        name = b[pos:pos + 4]
        pos += 4
        if name == b'ENDT':
            return tags, pos
        (n,) = struct.unpack_from('<H', b, pos)
        pos += 2
        tags.append((name, b[pos:pos + n]))
        pos += n
    raise ValueError('unterminated tag list')


def parse(b):
    if b[:4] != b'MTLB':
        raise ValueError('not a metallib')
    (size,) = struct.unpack_from('<Q', b, 0x10)
    if size != len(b):
        raise ValueError(f'file size field {size} != {len(b)}')
    fl_off, fl_size, pub_off, pub_size, prv_off, prv_size, bc_off, bc_size = struct.unpack_from('<8Q', b, 0x18)
    (count,) = struct.unpack_from('<I', b, fl_off)
    pos = fl_off + 4
    funcs = []
    for _ in range(count):
        (fsize,) = struct.unpack_from('<I', b, pos)
        tags, end = read_tags(b, pos + 4, pos + fsize)
        if end != pos + fsize:
            raise ValueError('function entry size mismatch')
        d = dict(tags)
        name = d[b'NAME'].rstrip(b'\0').decode()
        pub, prv, bc = struct.unpack('<3Q', d[b'OFFT'])
        (mdsz,) = struct.unpack('<Q', d[b'MDSZ'])
        funcs.append({'name': name, 'tags': tags, 'pub': pub, 'prv': prv, 'bc': bc, 'bcsize': mdsz})
        pos += fsize
    # The size field does not count the u32 function count.
    if pos != fl_off + 4 + fl_size:
        raise ValueError('function list size mismatch')
    ext_tags, ext_end = read_tags(b, pos, pub_off)
    if ext_tags or ext_end != pub_off:
        raise ValueError('header extension with tags (not supported)')
    sec = {'pub': (pub_off, pub_size), 'prv': (prv_off, prv_size), 'bc': (bc_off, bc_size)}
    return funcs, sec


def md_entry(b, sec, off):
    """A public/private metadata entry: u32 size, then that many bytes (tags ending in ENDT)."""
    start = sec[0] + off
    (n,) = struct.unpack_from('<I', b, start)
    return b[start:start + 4 + n]


def write_tags(tags):
    out = b''
    for name, data in tags:
        out += name + struct.pack('<H', len(data)) + data
    return out + b'ENDT'


def extract(b, names):
    funcs, sec = parse(b)
    pick = [f for f in funcs if f['name'] in names]
    missing = set(names) - {f['name'] for f in pick}
    if missing:
        raise ValueError(f'not in the library: {sorted(missing)}')
    pubs, prvs, bcs = b'', b'', b''
    entries = []
    seen_bc = {}
    for f in pick:
        pub = md_entry(b, sec['pub'], f['pub'])
        prv = md_entry(b, sec['prv'], f['prv'])
        if f['bc'] not in seen_bc:
            seen_bc[f['bc']] = len(bcs)
            bcs += b[sec['bc'][0] + f['bc']: sec['bc'][0] + f['bc'] + f['bcsize']]
        offt = struct.pack('<3Q', len(pubs), len(prvs), seen_bc[f['bc']])
        pubs += pub
        prvs += prv
        tags = [(n, offt if n == b'OFFT' else d) for n, d in f['tags']]
        body = write_tags(tags)
        entries.append(struct.pack('<I', 4 + len(body)) + body)
    flist = struct.pack('<I', len(entries)) + b''.join(entries)
    ext = b'ENDT'
    fl_off = 0x58
    pub_off = fl_off + len(flist) + len(ext)
    prv_off = pub_off + len(pubs)
    bc_off = prv_off + len(prvs)
    total = bc_off + len(bcs)
    hdr = bytearray(b[:0x10]) + struct.pack('<Q', total)
    hdr += struct.pack('<8Q', fl_off, len(flist) - 4, pub_off, len(pubs), prv_off, len(prvs), bc_off, len(bcs))
    assert len(hdr) == 0x58
    out = bytes(hdr) + flist + ext + pubs + prvs + bcs
    assert len(out) == total
    return out


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == '--list':
        funcs, _ = parse(open(sys.argv[2], 'rb').read())
        for f in funcs:
            print(f"{f['name']} pub={f['pub']} prv={f['prv']} bc={f['bc']} bcsize={f['bcsize']}")
        return
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    b = open(sys.argv[1], 'rb').read()
    out = extract(b, sys.argv[3:])
    # Self-check: the result parses and holds exactly the requested functions.
    funcs, _ = parse(out)
    assert sorted(f['name'] for f in funcs) == sorted(set(sys.argv[3:]))
    open(sys.argv[2], 'wb').write(out)
    print(f'{sys.argv[2]}: {len(funcs)} functions, {len(out)} bytes')


if __name__ == '__main__':
    main()
