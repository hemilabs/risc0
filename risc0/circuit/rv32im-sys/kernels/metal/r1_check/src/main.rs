// Copyright 2026 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! R1 Metal-vs-CPU check (plan 4.2, item R1). Runs, on the system GPU:
//! - the crate metallib's `injectCycleCols`, `finalizeZeroizeAccum`, `seq_fwd_probe` and
//!   `seq_rev_probe` (kernels/metal/misc.metal) on synthetic buffers, against a CPU model of the
//!   CUDA kernels (kernels/cuda/ffi.cu);
//! - `selftest.metal` (this directory), compiled here with and without -DEF_WIDE, which checks
//!   ef_fp.h/ef_fpext.h against fp.h/fpext.h and exercises witgen.h/buffers.h/tables.h the way
//!   translated code will, against a CPU model.
//!
//! Prints one `PASS`/`FAIL` line per case and exits non-zero on any failure.

// objc 0.2's msg_send! expands to cfg(feature = "cargo-clippy").
#![allow(unexpected_cfgs)]

use std::{path::PathBuf, process::Command, time::SystemTime};

use objc::{msg_send, sel, sel_impl};

use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLCommandBufferStatus,
    MTLResourceOptions, MTLSize,
};

// ---------------------------------------------------------------------------------------------
// CPU model of fp.h (raw Montgomery words, R = 2^32)
// ---------------------------------------------------------------------------------------------

const P: u64 = 15 * (1 << 27) + 1;
const INVALID: u32 = 0xffff_ffff;

fn pow_mod(mut b: u64, mut e: u64) -> u64 {
    let mut r = 1u64;
    b %= P;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % P;
        }
        b = b * b % P;
        e >>= 1;
    }
    r
}

/// `Fp(x)`: raw word of x mod P.
fn enc(x: u64) -> u32 {
    (((x % P) << 32) % P) as u32
}
fn add(a: u32, b: u32) -> u32 {
    ((a as u64 + b as u64) % P) as u32
}
fn sub(a: u32, b: u32) -> u32 {
    ((a as u64 + P - b as u64) % P) as u32
}
/// Raw a * raw b (Montgomery product).
fn mul(a: u32, b: u32) -> u32 {
    let rinv = pow_mod(1 << 32, P - 2);
    ((a as u64 * b as u64 % P) * rinv % P) as u32
}
/// Raw word of the field inverse of the value whose raw word is `a` (0 -> 0).
fn inv_raw(a: u32) -> u32 {
    // value = a * R^-1; inverse value = R / a; raw = R^2 / a.
    if a == 0 {
        return 0;
    }
    let r = (1u64 << 32) % P;
    (r * r % P * pow_mod(a as u64, P - 2) % P) as u32
}
fn ext_mul(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    let nb = enc(P - 11);
    let s = |xs: &[(u32, u32)]| xs.iter().fold(0, |acc, &(x, y)| add(acc, mul(x, y)));
    [
        add(
            mul(a[0], b[0]),
            mul(nb, s(&[(a[1], b[3]), (a[2], b[2]), (a[3], b[1])])),
        ),
        add(
            s(&[(a[0], b[1]), (a[1], b[0])]),
            mul(nb, s(&[(a[2], b[3]), (a[3], b[2])])),
        ),
        add(
            s(&[(a[0], b[2]), (a[1], b[1]), (a[2], b[0])]),
            mul(nb, mul(a[3], b[3])),
        ),
        s(&[(a[0], b[3]), (a[1], b[2]), (a[2], b[1]), (a[3], b[0])]),
    ]
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn u32(&mut self) -> u32 {
        (self.next() >> 16) as u32
    }
    fn fp(&mut self) -> u32 {
        (self.next() % P) as u32
    }
}

// ---------------------------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------------------------

struct Gpu {
    dev: Device,
    queue: CommandQueue,
}

enum Arg<'a> {
    Buf(&'a Buffer),
    Bytes(&'a [u8]),
}

fn bytes_of<T: Copy>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, size_of::<T>()) }
}

impl Gpu {
    fn buf<T: Copy>(&self, data: &[T]) -> Buffer {
        let len = (size_of_val(data) as u64).max(4);
        let b = self
            .dev
            .new_buffer(len, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                b.contents() as *mut u8,
                size_of_val(data),
            )
        };
        b
    }

    fn pso(&self, lib: &Library, name: &str) -> ComputePipelineState {
        let f = lib
            .get_function(name, None)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        self.dev
            .new_compute_pipeline_state_with_function(&f)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    /// Encode `dispatches` (args, threads) in order into one encoder of one command buffer, then
    /// wait. Returns the GPU time in microseconds.
    fn run(&self, pso: &ComputePipelineState, dispatches: &[(Vec<Arg>, u64)]) -> f64 {
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        let tg = pso.max_total_threads_per_threadgroup().min(256);
        for (args, threads) in dispatches {
            for (i, a) in args.iter().enumerate() {
                match a {
                    Arg::Buf(b) => enc.set_buffer(i as u64, Some(b), 0),
                    Arg::Bytes(d) => {
                        enc.set_bytes(i as u64, d.len() as u64, d.as_ptr() as *const _)
                    }
                }
            }
            enc.dispatch_threads(
                MTLSize::new(*threads, 1, 1),
                MTLSize::new(tg.min(*threads), 1, 1),
            );
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        assert_eq!(
            cb.status(),
            MTLCommandBufferStatus::Completed,
            "command buffer failed"
        );
        // metal-rs 0.29 has no accessor for GPUStartTime/GPUEndTime (seconds).
        let obj = cb as *const _ as *mut objc::runtime::Object;
        let (start, end): (f64, f64) =
            unsafe { (msg_send![obj, GPUStartTime], msg_send![obj, GPUEndTime]) };
        (end - start) * 1e6
    }
}

fn read<T: Copy>(b: &Buffer, n: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(b.contents() as *const T, v.as_mut_ptr(), n);
        v.set_len(n);
    }
    v
}

// ---------------------------------------------------------------------------------------------
// Param mirrors (misc.metal / seq_exec.h / selftest.metal)
// ---------------------------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct InjectParams {
    n: u32,
    rows: u32,
    c_cycle: u32,
    c_pc_low: u32,
    c_pc_high: u32,
    c_state: u32,
    c_mode: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FinalizeAccumParams {
    rows: u32,
    cols: u32,
    last_cycle: u32,
    user_accum_split: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SeqParams {
    begin: u32,
    end: u32,
    split: u32,
    last_cycle: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SeqProbeParams {
    rows: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct WitParams {
    rows: u32,
    use_pre: u32,
    zero_back: u32,
    _pad: u32,
}

/// RawPreflightCycle (rv32im-sys/src/lib.rs), 36 bytes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Cycle {
    state: u32,
    pc: u32,
    major: u8,
    minor: u8,
    machine_mode: u8,
    padding: u8,
    user_cycle: u32,
    txn_idx: u32,
    paging_idx: u32,
    bigint_idx: u32,
    diff_count: [u32; 2],
}

const SEQ_CHUNK: u32 = 4096; // RV32IM_SEQ_CHUNK

// ---------------------------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------------------------

struct Report {
    failures: usize,
}
impl Report {
    fn check(&mut self, name: &str, ok: bool, detail: String) {
        if ok {
            println!("PASS {name} {detail}");
        } else {
            println!("FAIL {name} {detail}");
            self.failures += 1;
        }
    }
}

fn first_diff(a: &[u32], b: &[u32]) -> String {
    match a.iter().zip(b).position(|(x, y)| x != y) {
        None => "identical".into(),
        Some(i) => format!("first diff at {i}: gpu {:#x} cpu {:#x}", a[i], b[i]),
    }
}

fn inject(gpu: &Gpu, lib: &Library, rep: &mut Report, rng: &mut Rng) {
    let pso = gpu.pso(lib, "injectCycleCols");
    for (rows, n) in [(1024u32, 1024u32), (1000, 977), (1 << 16, (1 << 16) - 3)] {
        let cols = 8u32;
        let data: Vec<u32> = (0..rows * cols).map(|_| rng.fp()).collect();
        let cycles: Vec<Cycle> = (0..n)
            .map(|_| Cycle {
                state: rng.u32(),
                pc: rng.u32(),
                major: rng.u32() as u8,
                minor: rng.u32() as u8,
                machine_mode: rng.u32() as u8,
                txn_idx: rng.u32(),
                ..Default::default()
            })
            .collect();
        let p = InjectParams {
            n,
            rows,
            c_cycle: 6,
            c_pc_low: 0,
            c_pc_high: 3,
            c_state: 7,
            c_mode: 2,
            _pad: 0,
        };
        let mut want = data.clone();
        for (r, c) in cycles.iter().enumerate() {
            let at = |col: u32| (col * rows) as usize + r;
            want[at(p.c_cycle)] = enc(r as u64);
            want[at(p.c_pc_low)] = enc((c.pc & 0xffff) as u64);
            want[at(p.c_pc_high)] = enc((c.pc >> 16) as u64);
            want[at(p.c_state)] = enc(c.state as u64);
            want[at(p.c_mode)] = enc(c.machine_mode as u64);
        }
        let bd = gpu.buf(&data);
        let bc = gpu.buf(&cycles);
        // Dispatch over all rows: threads >= n must leave their rows alone.
        let us = gpu.run(
            &pso,
            &[(
                vec![Arg::Buf(&bd), Arg::Buf(&bc), Arg::Bytes(bytes_of(&p))],
                rows as u64,
            )],
        );
        let got: Vec<u32> = read(&bd, want.len());
        rep.check(
            &format!("injectCycleCols rows={rows} n={n}"),
            got == want,
            format!("({}; {us:.0} us)", first_diff(&got, &want)),
        );
    }
}

fn finalize_cpu(buf: &[u32], rows: u32, cols: u32, last: u32, split: u32) -> Vec<u32> {
    let mut out = buf.to_vec();
    let idx = |r: u32, c: u32| (c * rows + r) as usize;
    let z = |v: u32| if v == INVALID { 0 } else { v };
    let mc = if cols >= split + 4 {
        (cols - split) / 4
    } else {
        0
    };
    let fin_end = split + mc.saturating_sub(1) * 4;
    for cycle in 0..rows {
        if cycle < last && mc > 0 {
            let back1 = (cycle + last - 1) % last;
            let prev: Vec<u32> = (0..4).map(|k| buf[idx(back1, cols - 4 + k)]).collect();
            for c in split..fin_end {
                out[idx(cycle, c)] = add(z(buf[idx(cycle, c)]), prev[((c - split) % 4) as usize]);
            }
            for c in (0..split).chain(fin_end..cols) {
                out[idx(cycle, c)] = z(buf[idx(cycle, c)]);
            }
        } else {
            for c in 0..cols {
                out[idx(cycle, c)] = z(buf[idx(cycle, c)]);
            }
        }
    }
    out
}

fn finalize(gpu: &Gpu, lib: &Library, rep: &mut Report, rng: &mut Rng) {
    let pso = gpu.pso(lib, "finalizeZeroizeAccum");
    let split = 23u32;
    for (rows, last, cols) in [
        (1024u32, 1024u32, 103u32),
        (1024, 1000, 103),
        (1000, 999, 103),
        (64, 1, 103),
        (1 << 16, (1 << 16) - 5, 103),
        (256, 200, 27), // one machine column group: nothing to finalize, zeroize only
    ] {
        let mut buf: Vec<u32> = (0..rows * cols)
            .map(|_| {
                if rng.u32().is_multiple_of(4) {
                    INVALID
                } else {
                    rng.fp()
                }
            })
            .collect();
        // The last 4 columns of the segment's rows are scan outputs: never INVALID.
        for r in 0..last {
            for k in 0..4 {
                buf[((cols - 4 + k) * rows + r) as usize] = rng.fp();
            }
        }
        let want = finalize_cpu(&buf, rows, cols, last, split);
        let p = FinalizeAccumParams {
            rows,
            cols,
            last_cycle: last,
            user_accum_split: split,
        };
        let b = gpu.buf(&buf);
        let us = gpu.run(
            &pso,
            &[(vec![Arg::Buf(&b), Arg::Bytes(bytes_of(&p))], rows as u64)],
        );
        let got: Vec<u32> = read(&b, want.len());
        let no_invalid = !got.contains(&INVALID);
        rep.check(
            &format!("finalizeZeroizeAccum rows={rows} last={last} cols={cols}"),
            got == want && no_invalid,
            format!("({}; {us:.0} us)", first_diff(&got, &want)),
        );
    }
}

fn seq(gpu: &Gpu, lib: &Library, rep: &mut Report, rng: &mut Rng) {
    for rev in [false, true] {
        let name = if rev {
            "seq_rev_probe"
        } else {
            "seq_fwd_probe"
        };
        let pso = gpu.pso(lib, name);
        for (rows, last, split) in [
            (10000u32, 9500u32, 3000u32),
            (4096, 4096, 0),
            (4096, 4096, 4096),
            (20000, 20000, 8193),
        ] {
            let mut data: Vec<u32> = (0..rows).map(|_| rng.fp()).collect();
            data.extend(std::iter::repeat_n(0, rows as usize));
            // CPU: CUDA's fwd_stepExec / rev_stepExec order.
            let order: Vec<u32> = if rev {
                (0..split).rev().chain((split..last).rev()).collect()
            } else {
                (0..last).collect()
            };
            let mut want = data.clone();
            for (seqno, &c) in order.iter().enumerate() {
                want[(rows + c) as usize] = enc(seqno as u64);
                let prev = want[((c + rows - 1) % rows) as usize];
                want[c as usize] = add(mul(prev, enc(3)), enc(c as u64 + 1));
            }
            let bd = gpu.buf(&data);
            let bctr = gpu.buf(&[0u32]);
            let q = SeqProbeParams { rows, _pad: [0; 3] };
            let params: Vec<SeqParams> = (0..last)
                .step_by(SEQ_CHUNK as usize)
                .map(|b| SeqParams {
                    begin: b,
                    end: (b + SEQ_CHUNK).min(last),
                    split,
                    last_cycle: last,
                })
                .collect();
            let dispatches: Vec<_> = params
                .iter()
                .map(|p| {
                    (
                        vec![
                            Arg::Buf(&bd),
                            Arg::Buf(&bctr),
                            Arg::Bytes(bytes_of(p)),
                            Arg::Bytes(bytes_of(&q)),
                        ],
                        1u64,
                    )
                })
                .collect();
            let us = gpu.run(&pso, &dispatches);
            let got: Vec<u32> = read(&bd, want.len());
            let ctr: Vec<u32> = read(&bctr, 1);
            rep.check(
                &format!(
                    "{name} rows={rows} last={last} split={split} chunks={}",
                    params.len()
                ),
                got == want && ctr[0] == last,
                format!(
                    "(counter {}; {}; {us:.0} us)",
                    ctr[0],
                    first_diff(&got, &want)
                ),
            );
        }
    }
}

fn ef(gpu: &Gpu, lib: &Library, tag: &str, rep: &mut Report, rng: &mut Rng) {
    let pso = gpu.pso(lib, "ef_selftest");
    let n = 1u32 << 16;
    let edge = [
        0u32,
        1,
        (P - 1) as u32,
        enc(1),
        enc(P - 1),
        enc(11),
        enc(P - 11),
        (P - 2) as u32,
    ];
    let a: Vec<u32> = (0..n)
        .map(|i| {
            if (i as usize) < edge.len() {
                edge[i as usize]
            } else {
                rng.fp()
            }
        })
        .collect();
    let b: Vec<u32> = (0..n)
        .map(|i| {
            if (i as usize) < edge.len() * edge.len() {
                edge[i as usize % edge.len()]
            } else {
                rng.fp()
            }
        })
        .collect();
    let ba = gpu.buf(&a);
    let bb = gpu.buf(&b);
    let bo = gpu.buf(&vec![0u32; n as usize * 16]);
    let bbad = gpu.buf(&[0u32]);
    let us = gpu.run(
        &pso,
        &[(
            vec![
                Arg::Buf(&ba),
                Arg::Buf(&bb),
                Arg::Buf(&bo),
                Arg::Buf(&bbad),
                Arg::Bytes(bytes_of(&n)),
            ],
            n as u64,
        )],
    );
    let out: Vec<u32> = read(&bo, n as usize * 16);
    let bad: Vec<u32> = read(&bbad, 1);
    let mut cpu_bad = 0usize;
    let mut first = String::new();
    for i in 0..n as usize {
        let o = &out[i * 16..i * 16 + 16];
        let (x, y) = (a[i], b[i]);
        let j = (i + 1) % n as usize;
        let k = (i + 2) % n as usize;
        let l = (i + 3) % n as usize;
        let ex = [x, a[j], a[k], a[l]];
        let ey = [y, b[j], b[k], b[l]];
        let em = ext_mul(ex, ey);
        let mut ok = o[0] == add(x, y)
            && o[1] == sub(x, y)
            && o[2] == mul(x, y)
            && o[3] == sub(0, x)
            && o[4] == inv_raw(x)
            && o[5..9] == em;
        // inv_ext_chain: ex * inv == 1 (or 0 for ex == 0).
        let ei = [o[9], o[10], o[11], o[12]];
        let one = if ex == [0; 4] {
            [0; 4]
        } else {
            [enc(1), 0, 0, 0]
        };
        ok &= ext_mul(ex, ei) == one;
        if !ok {
            cpu_bad += 1;
            if first.is_empty() {
                first = format!("first bad i={i} x={x:#x} y={y:#x} gpu={o:x?}");
            }
        }
    }
    rep.check(
        &format!("ef_selftest{tag} n={n}"),
        bad[0] == 0 && cpu_bad == 0,
        format!(
            "(gpu-vs-fp.h mismatches {}, cpu mismatches {cpu_bad} {first}; {us:.0} us)",
            bad[0]
        ),
    );
}

fn witgen(gpu: &Gpu, lib: &Library, rep: &mut Report, rng: &mut Rng) {
    let pso = gpu.pso(lib, "witgen_selftest");
    for rows in [1024u32, 1000] {
        for use_pre in [0u32, 1] {
            for zero_back in [0u32, 2] {
                let data: Vec<u32> = (0..rows * 4).map(|_| rng.fp()).collect();
                let pre: Vec<u32> = (0..rows * 4).map(|_| rng.fp()).collect();
                let global: Vec<u32> = (0..4).map(|_| rng.fp()).collect();
                let cycles = vec![Cycle::default(); rows as usize];
                let txns = vec![0u32; 5];
                let p = WitParams {
                    rows,
                    use_pre,
                    zero_back,
                    _pad: 0,
                };

                // CPU model.
                let load = |cyc: u32, col: u32, back: u32| -> u32 {
                    if zero_back != 0 && col > zero_back && back > 0 {
                        return 0;
                    }
                    let br = (rows + cyc - back) % rows;
                    let src = if back > 0 && use_pre != 0 {
                        &pre
                    } else {
                        &data
                    };
                    src[(col * rows + br) as usize]
                };
                let mut want = vec![0u32; (rows * 8) as usize];
                let mut t8 = vec![0u32; 256];
                let mut t16 = vec![0u32; 65536];
                for c in 0..rows {
                    let v0 = load(c, 0, 0);
                    let v1 = load(c, 1, 1);
                    let v2 = load(c, 3, 5);
                    let v3 = global[2];
                    let d = [add(v0, v0), add(v1, v1), add(enc(c as u64), enc(c as u64))];
                    let v4 = d.iter().fold(0, |acc, &x| add(mul(acc, enc(3)), x));
                    let v5 = (0..4).fold(0, |acc, i| add(acc, mul(enc(1), load(c, i, 2))));
                    let e: Vec<u32> = (0..4).map(|i| load(c, i, 3)).collect();
                    let v6 = load(c, 2, 0);
                    let term = mul(inv_raw(enc(c as u64 + 1)), enc((c & 5) as u64));
                    let col = |k: u32| (k * rows + c) as usize;
                    want[col(0)] = add(v0, v1);
                    want[col(1)] = v2;
                    want[col(2)] = add(v3, v4);
                    want[col(3)] = add(add(v5, mul(v6, enc(1))), term);
                    for k in 0..4 {
                        want[col(4 + k)] = e[k as usize];
                    }
                    t8[(c & 255) as usize] += 1;
                    t16[((c * 7) & 0xffff) as usize] += 1;
                }
                let mut want_global = global.clone();
                want_global[3] = enc(77);

                let bd = gpu.buf(&data);
                let bp = gpu.buf(&pre);
                let bg = gpu.buf(&global);
                let bo = gpu.buf(&vec![0u32; (rows * 8) as usize]);
                let b8 = gpu.buf(&vec![0u32; 256]);
                let b16 = gpu.buf(&vec![0u32; 65536]);
                let bc = gpu.buf(&cycles);
                let bt = gpu.buf(&txns);
                let bf = gpu.buf(&[0u32]);
                gpu.run(
                    &pso,
                    &[(
                        vec![
                            Arg::Buf(&bd),
                            Arg::Buf(&bp),
                            Arg::Buf(&bg),
                            Arg::Buf(&bo),
                            Arg::Buf(&b8),
                            Arg::Buf(&b16),
                            Arg::Buf(&bc),
                            Arg::Buf(&bt),
                            Arg::Buf(&bf),
                            Arg::Bytes(bytes_of(&p)),
                        ],
                        rows as u64,
                    )],
                );
                let got: Vec<u32> = read(&bo, want.len());
                let got_global: Vec<u32> = read(&bg, 4);
                let got_data: Vec<u32> = read(&bd, data.len());
                let got8: Vec<u32> = read(&b8, 256);
                let got16: Vec<u32> = read(&b16, 65536);
                let fault: Vec<u32> = read(&bf, 1);
                let ok = got == want
                    && got_global == want_global
                    && got_data == data
                    && got8 == t8
                    && got16 == t16
                    && fault[0] == 0x3;
                rep.check(
                    &format!("witgen_selftest rows={rows} use_pre={use_pre} zero_back={zero_back}"),
                    ok,
                    format!(
                        "(out {}; global {}; tables {}; fault {:#x})",
                        first_diff(&got, &want),
                        if got_global == want_global {
                            "ok"
                        } else {
                            "DIFF"
                        },
                        if got8 == t8 && got16 == t16 {
                            "ok"
                        } else {
                            "DIFF"
                        },
                        fault[0]
                    ),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Build / locate the metallibs
// ---------------------------------------------------------------------------------------------

fn here() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// <repo>/ from <repo>/risc0/circuit/rv32im-sys/kernels/metal/r1_check.
fn repo_root() -> PathBuf {
    here().ancestors().nth(6).unwrap().to_path_buf()
}

fn find_crate_metallib() -> PathBuf {
    let mut best: Option<(SystemTime, PathBuf)> = None;
    let target = repo_root().join("target");
    for profile in std::fs::read_dir(&target)
        .unwrap_or_else(|_| panic!("no {target:?}"))
        .flatten()
    {
        let Ok(builds) = std::fs::read_dir(profile.path().join("build")) else {
            continue;
        };
        for b in builds.flatten() {
            if !b
                .file_name()
                .to_string_lossy()
                .starts_with("risc0-circuit-rv32im-sys-")
            {
                continue;
            }
            let lib = b.path().join("out/metal_kernels_rv32im.metallib");
            let Ok(meta) = lib.metadata() else { continue };
            let Ok(m) = meta.modified() else { continue };
            if meta.len() > 0 && best.as_ref().is_none_or(|(t, _)| m > *t) {
                best = Some((m, lib));
            }
        }
    }
    best.map(|(_, p)| p)
        .expect("no metal_kernels_rv32im.metallib: run `cargo build -p risc0-circuit-rv32im-sys --features metal`")
}

fn build_selftest(defines: &[&str], tag: &str) -> Vec<u8> {
    let out = std::env::temp_dir().join("risc0-r1-check");
    std::fs::create_dir_all(&out).unwrap();
    let air = out.join(format!("selftest{tag}.air"));
    let lib = out.join(format!("selftest{tag}.metallib"));
    let inc_sys = repo_root().join("risc0/build_kernel/kernels/metal");
    let inc_rv = here().parent().unwrap().to_path_buf();
    let o = Command::new("xcrun")
        .args(["metal", "-std=metal3.2", "-O2", "-Werror"])
        .args(defines)
        .arg("-I")
        .arg(&inc_sys)
        .arg("-I")
        .arg(&inc_rv)
        .arg("-c")
        .arg(here().join("selftest.metal"))
        .arg("-o")
        .arg(&air)
        .output()
        .expect("xcrun not found: source /Users/user/zkminer/env.sh");
    assert!(
        o.status.success(),
        "selftest compile: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    let o = Command::new("xcrun")
        .arg("metallib")
        .arg(&air)
        .arg("-o")
        .arg(&lib)
        .output()
        .unwrap();
    assert!(
        o.status.success(),
        "metallib: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    std::fs::read(lib).unwrap()
}

fn main() {
    let dev = Device::system_default().expect("no Metal device");
    println!("device: {}", dev.name());
    let gpu = Gpu {
        queue: dev.new_command_queue(),
        dev,
    };
    let mut rep = Report { failures: 0 };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);

    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(find_crate_metallib);
    println!("crate metallib: {}", path.display());
    let crate_lib = gpu
        .dev
        .new_library_with_data(&std::fs::read(&path).unwrap())
        .unwrap();
    let mut names = crate_lib.function_names();
    names.sort();
    println!("crate metallib functions: {names:?}");
    inject(&gpu, &crate_lib, &mut rep, &mut rng);
    finalize(&gpu, &crate_lib, &mut rep, &mut rng);
    seq(&gpu, &crate_lib, &mut rep, &mut rng);

    for (defines, tag) in [(&[][..], ""), (&["-DEF_WIDE"][..], " EF_WIDE")] {
        let lib = gpu
            .dev
            .new_library_with_data(&build_selftest(defines, tag.trim()))
            .unwrap();
        ef(&gpu, &lib, tag, &mut rep, &mut rng);
        if tag.is_empty() {
            witgen(&gpu, &lib, &mut rep, &mut rng);
        }
    }

    if rep.failures > 0 {
        println!("{} FAILED", rep.failures);
        std::process::exit(1);
    }
    println!("all passed");
}
