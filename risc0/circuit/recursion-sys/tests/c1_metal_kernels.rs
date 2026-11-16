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

//! C1 (docs/metal-port/PLAN.md 4.3): GPU-vs-CPU checks of the hand-written recursion kernels in
//! `kernels/metal/rec_kernels.metal` (as embedded in [METAL_LIB]) and of the extern handlers in
//! `kernels/metal/extern.h` (through a probe kernel compiled at run time from the headers).
//!
//! This crate has no `metal`/`risc0-zkp` dev-dependency (and C1 may not edit Cargo.toml), so the
//! test drives Metal through a minimal Objective-C runtime binding (`objc_msgSend`) of the
//! handful of calls it needs. Run with
//! `cargo test -p risc0-circuit-recursion-sys --features metal --test c1_metal_kernels`.

#![cfg(all(feature = "metal", target_os = "macos"))]
#![allow(clippy::needless_range_loop)]

use std::ffi::{c_char, c_void, CStr};

use risc0_circuit_recursion_sys::{METAL_KERNEL_NAMES, METAL_LIB};
use risc0_core::field::{
    baby_bear::{BabyBearElem as Elem, BabyBearExtElem as ExtElem, P},
    Elem as _,
};

// ------------------------------------------------------------------------------------------------
// Minimal Metal binding.

type Id = *mut c_void;
type Sel = *const c_void;

#[link(name = "objc")]
extern "C" {
    fn objc_msgSend();
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_getClass(name: *const c_char) -> Id;
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

#[link(name = "Metal", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> Id;
}

#[link(name = "Foundation", kind = "framework")]
extern "C" {}

extern "C" {
    // libdispatch (libSystem); a NULL destructor (DISPATCH_DATA_DESTRUCTOR_DEFAULT) copies.
    fn dispatch_data_create(
        buffer: *const c_void,
        size: usize,
        queue: *mut c_void,
        destructor: *mut c_void,
    ) -> Id;
}

fn sel(name: &str) -> Sel {
    debug_assert!(name.ends_with('\0'));
    unsafe { sel_registerName(name.as_ptr() as *const c_char) }
}

/// `msg!(Ret; obj, "selector:with:", a => TypeA, b => TypeB)`
macro_rules! msg {
    ($ret:ty; $obj:expr, $sel:literal $(, $a:expr => $t:ty)* $(,)?) => {{
        #[allow(clippy::missing_transmute_annotations)]
        let f: unsafe extern "C" fn(Id, Sel $(, $t)*) -> $ret =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f($obj, sel(concat!($sel, "\0")) $(, $a)*)
    }};
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MTLSize {
    width: u64,
    height: u64,
    depth: u64,
}

fn size3(width: u64, height: u64, depth: u64) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}

fn nsstring(s: &str) -> Id {
    let c = std::ffi::CString::new(s).unwrap();
    unsafe {
        msg!(Id; objc_getClass(c"NSString".as_ptr()), "stringWithUTF8String:",
             c.as_ptr() => *const c_char)
    }
}

fn describe(err: Id) -> String {
    if err.is_null() {
        return "(no error object)".into();
    }
    unsafe {
        let s: Id = msg!(Id; err, "localizedDescription");
        let p: *const c_char = msg!(*const c_char; s, "UTF8String");
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

struct Buffer {
    obj: Id,
    words: usize,
}

impl Buffer {
    fn ptr(&self) -> *mut u32 {
        unsafe { msg!(*mut c_void; self.obj, "contents") as *mut u32 }
    }
    fn read(&self) -> Vec<u32> {
        unsafe { std::slice::from_raw_parts(self.ptr(), self.words).to_vec() }
    }
    fn write(&self, data: &[u32]) {
        assert!(data.len() <= self.words);
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr(), data.len()) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { msg!((); self.obj, "release") }
    }
}

enum Arg<'a> {
    Buf(&'a Buffer),
    U32(u32),
    Bytes(Vec<u8>),
}

enum Grid {
    /// dispatchThreads (non-uniform), threadgroup = min(256, pso max).
    Threads(u64),
    Threads2D(u64, u64),
    /// dispatchThreadgroups(groups, threads).
    Groups(u64, u64),
}

struct Gpu {
    device: Id,
    queue: Id,
    library: Id,
}

// The Metal objects are thread-safe; each test makes its own Gpu anyway.
unsafe impl Send for Gpu {}

impl Gpu {
    fn new() -> Self {
        Self::with_library(|device| unsafe {
            let data = dispatch_data_create(
                METAL_LIB.as_ptr() as *const c_void,
                METAL_LIB.len(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            let mut err: Id = std::ptr::null_mut();
            let lib =
                msg!(Id; device, "newLibraryWithData:error:", data => Id, &mut err => *mut Id);
            assert!(
                !lib.is_null(),
                "newLibraryWithData failed: {}",
                describe(err)
            );
            lib
        })
    }

    /// Compile `source` at run time with `-std=metal3.2` (MTLLanguageVersion3_2).
    fn from_source(source: &str) -> Self {
        Self::with_library(|device| unsafe {
            let opts_cls = objc_getClass(c"MTLCompileOptions".as_ptr());
            let opts: Id = msg!(Id; opts_cls, "new");
            msg!((); opts, "setLanguageVersion:", (3u64 << 16) | 2 => u64);
            let mut err: Id = std::ptr::null_mut();
            let lib = msg!(Id; device, "newLibraryWithSource:options:error:",
                           nsstring(source) => Id, opts => Id, &mut err => *mut Id);
            assert!(
                !lib.is_null(),
                "newLibraryWithSource failed: {}",
                describe(err)
            );
            lib
        })
    }

    fn with_library(make: impl FnOnce(Id) -> Id) -> Self {
        unsafe {
            let device = MTLCreateSystemDefaultDevice();
            assert!(!device.is_null(), "no Metal device");
            let queue = msg!(Id; device, "newCommandQueue");
            let library = make(device);
            Gpu {
                device,
                queue,
                library,
            }
        }
    }

    fn try_pipeline(&self, name: &str) -> Result<Id, String> {
        unsafe {
            let func = msg!(Id; self.library, "newFunctionWithName:", nsstring(name) => Id);
            if func.is_null() {
                return Err(format!("function {name} not in library"));
            }
            let mut err: Id = std::ptr::null_mut();
            let pso = msg!(Id; self.device, "newComputePipelineStateWithFunction:error:",
                           func => Id, &mut err => *mut Id);
            if pso.is_null() {
                return Err(format!("PSO {name}: {}", describe(err)));
            }
            Ok(pso)
        }
    }

    fn pipeline(&self, name: &str) -> Id {
        self.try_pipeline(name).unwrap()
    }

    fn buffer(&self, words: usize) -> Buffer {
        let bytes = (words.max(1) * 4) as u64;
        // MTLResourceStorageModeShared | hazard tracking default = 0.
        let obj = unsafe {
            msg!(Id; self.device, "newBufferWithLength:options:", bytes => u64, 0u64 => u64)
        };
        assert!(!obj.is_null());
        Buffer { obj, words }
    }

    fn buffer_from(&self, data: &[u32]) -> Buffer {
        let b = self.buffer(data.len());
        b.write(data);
        b
    }

    /// Encode every dispatch into one command buffer and one (serial) compute encoder, then commit
    /// and wait, like a MetalHal batch.
    fn run(&self, dispatches: &[(&str, Vec<Arg>, Grid)]) {
        unsafe {
            let pool = objc_autoreleasePoolPush();
            // Empty grids are skipped before anything is bound (the validation layer flags
            // unused bindings and empty encoders).
            let dispatches: Vec<_> = dispatches
                .iter()
                .filter(|d| match d.2 {
                    Grid::Threads(n) => n > 0,
                    Grid::Threads2D(x, y) => x > 0 && y > 0,
                    Grid::Groups(g, t) => g > 0 && t > 0,
                })
                .collect();
            if dispatches.is_empty() {
                objc_autoreleasePoolPop(pool);
                return;
            }
            let psos: Vec<Id> = dispatches.iter().map(|d| self.pipeline(d.0)).collect();
            let cb: Id = msg!(Id; self.queue, "commandBuffer");
            let enc: Id = msg!(Id; cb, "computeCommandEncoder");
            // Skip re-binding the same buffer at the same index (validation "redundant setting").
            let mut bound: Vec<Id> = Vec::new();
            for ((_, args, grid), pso) in dispatches.iter().map(|d| (d.0, &d.1, &d.2)).zip(&psos) {
                msg!((); enc, "setComputePipelineState:", *pso => Id);
                for (i, a) in args.iter().enumerate() {
                    if bound.len() <= i {
                        bound.resize(i + 1, std::ptr::null_mut());
                    }
                    let prev = std::mem::replace(&mut bound[i], std::ptr::null_mut());
                    let i = i as u64;
                    match a {
                        Arg::Buf(b) => {
                            bound[i as usize] = b.obj;
                            if prev != b.obj {
                                msg!((); enc, "setBuffer:offset:atIndex:",
                                     b.obj => Id, 0u64 => u64, i => u64)
                            }
                        }
                        Arg::U32(v) => msg!((); enc, "setBytes:length:atIndex:",
                                            v as *const u32 as *const c_void => *const c_void,
                                            4u64 => u64, i => u64),
                        Arg::Bytes(v) => msg!((); enc, "setBytes:length:atIndex:",
                                              v.as_ptr() as *const c_void => *const c_void,
                                              v.len() as u64 => u64, i => u64),
                    }
                }
                let max_tg: u64 = msg!(u64; *pso, "maxTotalThreadsPerThreadgroup");
                match *grid {
                    Grid::Threads(n) => {
                        msg!((); enc, "dispatchThreads:threadsPerThreadgroup:",
                             size3(n, 1, 1) => MTLSize, size3(256.min(max_tg), 1, 1) => MTLSize)
                    }
                    Grid::Threads2D(x, y) => {
                        msg!((); enc, "dispatchThreads:threadsPerThreadgroup:",
                             size3(x, y, 1) => MTLSize, size3(256.min(max_tg), 1, 1) => MTLSize)
                    }
                    Grid::Groups(groups, threads) => {
                        assert!(threads <= max_tg);
                        msg!((); enc, "dispatchThreadgroups:threadsPerThreadgroup:",
                             size3(groups, 1, 1) => MTLSize, size3(threads, 1, 1) => MTLSize)
                    }
                }
            }
            msg!((); enc, "endEncoding");
            msg!((); cb, "commit");
            msg!((); cb, "waitUntilCompleted");
            let status: u64 = msg!(u64; cb, "status");
            // MTLCommandBufferStatusCompleted = 4
            assert_eq!(
                status,
                4,
                "command buffer failed: {}",
                describe(msg!(Id; cb, "error"))
            );
            for pso in psos {
                msg!((); pso, "release");
            }
            objc_autoreleasePoolPop(pool);
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Host mirrors of the context.h params structs (sizes static_asserted there).

fn words_to_bytes(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[derive(Clone, Copy, Default)]
struct RecExecParams {
    total_cycles: u32,
    num_cycles: u32,
    num_woms: u32,
    num_iops: u32,
    num_wom_rows: u32,
}
impl RecExecParams {
    fn arg(&self) -> Arg<'static> {
        let w = [
            self.total_cycles,
            self.num_cycles,
            self.num_woms,
            self.num_iops,
            self.num_wom_rows,
            0,
            0,
            0,
        ];
        Arg::Bytes(words_to_bytes(&w)) // 32 bytes
    }
}

#[derive(Clone, Copy, Default)]
struct RecRowParams {
    log_rows: u32,
    live: u32,
    cols: u32,
    zk: u32,
    value: u32,
}
impl RecRowParams {
    fn arg(&self) -> Arg<'static> {
        let w = [
            self.log_rows,
            self.live,
            self.cols,
            self.zk,
            self.value,
            0,
            0,
            0,
        ];
        Arg::Bytes(words_to_bytes(&w)) // 32 bytes
    }
}

const K_MAX_WOM_ROWS_PER_CYCLE: usize = 9;
const K_SCAN_THREADS: u64 = 256;
const K_SCAN_CHUNK: u64 = 2048;
const INVALID: u32 = 0xffff_ffff;

// ------------------------------------------------------------------------------------------------
// Deterministic test data.

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
    fn elem(&mut self) -> Elem {
        Elem::new(self.below(P - 1) + 1)
    }
    fn ext(&mut self) -> ExtElem {
        ExtElem::new(self.elem(), self.elem(), self.elem(), self.elem())
    }
}

fn ext_words(v: &[ExtElem]) -> Vec<u32> {
    v.iter().flat_map(|e| e.to_u32_words()).collect()
}

fn raw(e: Elem) -> u32 {
    e.as_u32_montgomery()
}

// ------------------------------------------------------------------------------------------------
// Tests.

#[test]
fn c1_every_listed_kernel_builds_a_pipeline() {
    let gpu = Gpu::new();
    // The first 10 names are rec_kernels.metal's; C4 appended the translated step/eval_check
    // kernels, whose PSOs c4_metallib_smoke builds.
    let names = &METAL_KERNEL_NAMES[..10];
    assert_eq!(names.last(), Some(&"scatterHostChains"));
    for name in names {
        let pso = gpu.try_pipeline(name).unwrap();
        let max: u64 = unsafe { msg!(u64; pso, "maxTotalThreadsPerThreadgroup") };
        let tg: u64 = unsafe { msg!(u64; pso, "staticThreadgroupMemoryLength") };
        assert!(
            tg <= 32768,
            "{name}: static threadgroup memory {tg} > 32768"
        );
        if *name == "prefixProductsChunks" {
            assert_eq!(
                max, K_SCAN_THREADS,
                "max_total_threads_per_threadgroup(256)"
            );
            assert_eq!(tg, 4096);
        }
    }
    assert!(gpu.try_pipeline("noop").is_err(), "noop.metal was deleted");
}

/// Encode the accum-phase sequence the HAL will use (fillOneExt is not part of it here: the
/// input is random) as ffi.cu calcPrefixProducts does under SPPARK_NO_COOPERATIVE_LAUNCH.
fn prefix_trio(gpu: &Gpu, io: &Buffer, n: u32) {
    let nblocks = (n as u64).div_ceil(K_SCAN_CHUNK);
    let totals = gpu.buffer((nblocks as usize) * 4);
    let mut d = vec![(
        "prefixProductsChunks",
        vec![Arg::Buf(io), Arg::U32(n), Arg::Buf(&totals)],
        Grid::Groups(nblocks, K_SCAN_THREADS),
    )];
    if nblocks > 1 {
        d.push((
            "prefixProductsTotals",
            vec![Arg::Buf(&totals), Arg::U32(nblocks as u32)],
            Grid::Threads(1),
        ));
        d.push((
            "prefixProductsFixup",
            vec![Arg::Buf(io), Arg::U32(n), Arg::Buf(&totals)],
            Grid::Threads(n as u64 - K_SCAN_CHUNK),
        ));
    }
    gpu.run(&d);
}

#[test]
fn c1_prefix_products_match_cpu() {
    let gpu = Gpu::new();
    let mut rng = Rng(0xc1);
    for n in [1u32, 2, 255, 2047, 2048, 2049, 4096, 4097, 100_000, 1 << 18] {
        let input: Vec<ExtElem> = (0..n).map(|_| rng.ext()).collect();
        let mut expect = Vec::with_capacity(n as usize);
        let mut acc = ExtElem::ONE;
        for x in &input {
            acc *= *x;
            expect.push(acc);
        }
        // One guard element past n must stay untouched.
        let mut words = ext_words(&input);
        words.extend_from_slice(&[0xdead_beef; 4]);
        let io = gpu.buffer_from(&words);
        prefix_trio(&gpu, &io, n);
        let got = io.read();
        let want = ext_words(&expect);
        if let Some(i) = (0..want.len()).find(|&i| got[i] != want[i]) {
            panic!(
                "n = {n}: first mismatch at element {} (word {i}): got {:#x}, want {:#x}",
                i / 4,
                got[i],
                want[i]
            );
        }
        assert_eq!(
            &got[want.len()..],
            &[0xdead_beef; 4],
            "n = {n}: guard overwritten"
        );
    }
}

#[test]
fn c1_fill_one_ext_then_prefix_is_all_ones() {
    let gpu = Gpu::new();
    let one = ExtElem::ONE.to_u32_words();
    for n in [1u32, 1000, 2049, (1 << 18) + 3] {
        let io = gpu.buffer_from(&vec![0x1234_5678; (n as usize + 1) * 4]);
        gpu.run(&[(
            "fillOneExt",
            vec![Arg::Buf(&io), Arg::U32(n)],
            Grid::Threads(n as u64),
        )]);
        let got = io.read();
        for i in 0..n as usize {
            assert_eq!(&got[i * 4..i * 4 + 4], &one[..], "n = {n}: element {i}");
        }
        assert_eq!(&got[n as usize * 4..], &[0x1234_5678; 4], "n = {n}: guard");
        // The accum sequence: fill, (compute accum), prefix: ones stay ones.
        prefix_trio(&gpu, &io, n);
        let got = io.read();
        for i in 0..n as usize {
            assert_eq!(
                &got[i * 4..i * 4 + 4],
                &one[..],
                "n = {n}: prefix element {i}"
            );
        }
    }
}

/// CPU reference: kernels/cxx/ffi.cpp MachineContext::injectWomBacks, with the CUDA row layout
/// (`addr` is a raw Fp word).
fn inject_wom_backs_ref(
    data: &mut [u32],
    wom_index: &[u32],
    wom_rows: &[u32],
    num_cycles: usize,
    total: usize,
) {
    for cycle in 1..num_cycles {
        let idx = wom_index[cycle] as usize;
        for k in 0..5 {
            data[k * total + cycle - 1] = if idx != 0 {
                wom_rows[(idx - 1) * 5 + k]
            } else {
                raw(Elem::ZERO)
            };
        }
    }
}

#[test]
fn c1_inject_wom_backs_matches_cpu() {
    let gpu = Gpu::new();
    let mut rng = Rng(0x1e);
    for (num_cycles, total) in [(1usize, 4usize), (2, 4), (1000, 1024), (70_000, 1 << 17)] {
        // Per-cycle row counts (0..=9), exclusive-scanned as after the witgen sort.
        let counts: Vec<u32> = (0..num_cycles)
            .map(|_| match rng.below(4) {
                0 => 0,
                _ => rng.below(K_MAX_WOM_ROWS_PER_CYCLE as u32 + 1),
            })
            .collect();
        let mut wom_index = Vec::with_capacity(num_cycles);
        let mut acc = 0u32;
        for c in &counts {
            wom_index.push(acc);
            acc += c;
        }
        let num_wom_rows = num_cycles * K_MAX_WOM_ROWS_PER_CYCLE;
        let wom_rows: Vec<u32> = (0..num_wom_rows * 5).map(|_| raw(rng.elem())).collect();
        let cols = 6; // five touched columns plus one that must stay untouched
        let init: Vec<u32> = (0..cols * total).map(|i| 0xa000_0000 | i as u32).collect();
        let params = RecExecParams {
            total_cycles: total as u32,
            num_cycles: num_cycles as u32,
            num_wom_rows: num_wom_rows as u32,
            ..Default::default()
        };

        let data = gpu.buffer_from(&init);
        let wi = gpu.buffer_from(&wom_index);
        let wr = gpu.buffer_from(&wom_rows);
        let fault = gpu.buffer_from(&[0]);
        gpu.run(&[(
            "injectWomBacks",
            vec![
                Arg::Buf(&data),
                Arg::Buf(&wi),
                Arg::Buf(&wr),
                Arg::Buf(&fault),
                params.arg(),
            ],
            Grid::Threads(num_cycles as u64),
        )]);
        let mut want = init.clone();
        inject_wom_backs_ref(&mut want, &wom_index, &wom_rows, num_cycles, total);
        assert!(
            data.read() == want,
            "num_cycles = {num_cycles}: data differs"
        );
        assert_eq!(fault.read(), vec![0], "num_cycles = {num_cycles}: fault");
    }
}

#[test]
fn c1_inject_wom_backs_clamps_corrupt_index() {
    let gpu = Gpu::new();
    let (num_cycles, total) = (8usize, 8usize);
    let num_wom_rows = 4usize;
    let wom_rows: Vec<u32> = (0..num_wom_rows as u32 * 5)
        .map(|i| raw(Elem::new(i + 1)))
        .collect();
    // cycle 3 points one past the last row: in range (reads row 3); cycle 5 is corrupt.
    let wom_index = [0u32, 1, 2, 4, 4, 1000, 0, 3];
    let params = RecExecParams {
        total_cycles: total as u32,
        num_cycles: num_cycles as u32,
        num_wom_rows: num_wom_rows as u32,
        ..Default::default()
    };
    let data = gpu.buffer_from(&vec![7u32; 5 * total]);
    let wi = gpu.buffer_from(&wom_index);
    let wr = gpu.buffer_from(&wom_rows);
    let fault = gpu.buffer_from(&[0]);
    gpu.run(&[(
        "injectWomBacks",
        vec![
            Arg::Buf(&data),
            Arg::Buf(&wi),
            Arg::Buf(&wr),
            Arg::Buf(&fault),
            params.arg(),
        ],
        Grid::Threads(num_cycles as u64),
    )]);
    assert_eq!(fault.read(), vec![1]);
    let got = data.read();
    for k in 0..5 {
        assert_eq!(
            got[k * total + 4],
            raw(Elem::ZERO),
            "corrupt cycle writes zeros"
        );
        assert_eq!(
            got[k * total + 2],
            wom_rows[3 * 5 + k],
            "idx == numWomRows is valid"
        );
    }
}

// Row passes (ffi.cu recInitRows/recFillTail/recZeroizeRows).

fn row_passes_ref(
    log_rows: u32,
    live: u32,
    cols: u32,
    zk: u32,
    value: u32,
    writes: &[(usize, u32)],
) -> Vec<u32> {
    let rows = 1usize << log_rows;
    let mut buf = vec![0u32; cols as usize * rows];
    for c in 0..cols as usize {
        for r in 0..rows {
            buf[c * rows + r] = if (r as u32) < live { INVALID } else { 0 };
        }
        for r in rows - zk as usize..rows {
            buf[c * rows + r] = value;
        }
    }
    for &(i, v) in writes {
        buf[i] = v;
    }
    for c in 0..cols as usize {
        for r in 0..live as usize {
            if buf[c * rows + r] == INVALID {
                buf[c * rows + r] = 0;
            }
        }
    }
    buf
}

#[test]
fn c1_row_passes_match_cpu() {
    let gpu = Gpu::new();
    let mut rng = Rng(0x40);
    for (log_rows, live, cols, zk) in [
        (2u32, 0u32, 1u32, 0u32),
        (2, 3, 2, 1),
        (4, 13, 3, 2),
        (10, 1000, 5, 4),
        (12, 4093, 7, 4),
        (12, 4096 - 4, 3, 4),
    ] {
        let rows = 1usize << log_rows;
        let value = raw(rng.elem());
        let p = RecRowParams {
            log_rows,
            live,
            cols,
            zk,
            value,
        };
        // Some witgen writes in [0, live) between the init and the zeroize.
        let writes: Vec<(usize, u32)> = (0..(live as usize * cols as usize) / 3)
            .map(|_| {
                let c = rng.below(cols) as usize;
                let r = rng.below(live) as usize;
                (c * rows + r, raw(rng.elem()))
            })
            .collect();
        let buf = gpu.buffer_from(&vec![0x5555_5555; cols as usize * rows]);
        gpu.run(&[
            (
                "recInitRows",
                vec![Arg::Buf(&buf), p.arg()],
                Grid::Threads((cols as u64) << log_rows >> 2),
            ),
            (
                "recFillTail",
                vec![Arg::Buf(&buf), p.arg()],
                Grid::Threads((zk * cols) as u64),
            ),
        ]);
        let ptr = buf.ptr();
        for &(i, v) in &writes {
            unsafe { *ptr.add(i) = v };
        }
        gpu.run(&[(
            "recZeroizeRows",
            vec![Arg::Buf(&buf), p.arg()],
            Grid::Threads2D((live as u64).div_ceil(4), cols as u64),
        )]);
        let want = row_passes_ref(log_rows, live, cols, zk, value, &writes);
        assert!(
            buf.read() == want,
            "row passes differ: logRows {log_rows} live {live} cols {cols} zk {zk}"
        );
    }
}

// Host-chain kernels (ffi.cu markHostChains/scatterHostChains).

#[test]
fn c1_host_chain_kernels_match_cpu() {
    let gpu = Gpu::new();
    let mut rng = Rng(0x4c);
    let (num_cycles, total, data_cols) = (500usize, 512usize, 12usize);
    // markHostChains
    let cycles_init: Vec<u32> = (0..num_cycles * 2).map(|i| (i % 2) as u32).collect();
    let heads: Vec<u32> = vec![3, 10, 499, 77];
    let cycles = gpu.buffer_from(&cycles_init);
    let hb = gpu.buffer_from(&heads);
    let fault = gpu.buffer_from(&[0]);
    gpu.run(&[(
        "markHostChains",
        vec![
            Arg::Buf(&cycles),
            Arg::Buf(&hb),
            Arg::Buf(&fault),
            Arg::U32(heads.len() as u32),
            Arg::U32(num_cycles as u32),
        ],
        Grid::Threads(heads.len() as u64),
    )]);
    let mut want = cycles_init.clone();
    for &h in &heads {
        want[h as usize * 2 + 1] = 2;
    }
    assert!(cycles.read() == want);
    assert_eq!(fault.read(), vec![0]);
    // A head out of range faults and writes nothing.
    let bad = gpu.buffer_from(&[num_cycles as u32]);
    gpu.run(&[(
        "markHostChains",
        vec![
            Arg::Buf(&cycles),
            Arg::Buf(&bad),
            Arg::Buf(&fault),
            Arg::U32(1),
            Arg::U32(num_cycles as u32),
        ],
        Grid::Threads(1),
    )]);
    assert!(cycles.read() == want);
    assert_eq!(fault.read(), vec![1]);

    // scatterHostChains
    let rows_cycles: Vec<u32> = vec![4, 5, 6, 100, 101, 499];
    let n = rows_cycles.len();
    let rows: Vec<u32> = (0..n * data_cols).map(|_| raw(rng.elem())).collect();
    let wom_idx: Vec<u32> = (0..n as u32).map(|i| i * 3).collect();
    let src_wom: Vec<u32> = (0..n * K_MAX_WOM_ROWS_PER_CYCLE * 5)
        .map(|_| raw(rng.elem()))
        .collect();
    let data_init: Vec<u32> = (0..data_cols * total).map(|i| i as u32).collect();
    let wi_init: Vec<u32> = vec![0xeeee; num_cycles];
    let wr_init: Vec<u32> = vec![INVALID; num_cycles * K_MAX_WOM_ROWS_PER_CYCLE * 5];
    let data = gpu.buffer_from(&data_init);
    let wi = gpu.buffer_from(&wi_init);
    let wr = gpu.buffer_from(&wr_init);
    let cb = gpu.buffer_from(&rows_cycles);
    let rb = gpu.buffer_from(&rows);
    let wib = gpu.buffer_from(&wom_idx);
    let swb = gpu.buffer_from(&src_wom);
    let fault = gpu.buffer_from(&[0]);
    let params = words_to_bytes(&[n as u32, data_cols as u32, total as u32, num_cycles as u32]);
    gpu.run(&[(
        "scatterHostChains",
        vec![
            Arg::Buf(&data),
            Arg::Buf(&wi),
            Arg::Buf(&wr),
            Arg::Buf(&cb),
            Arg::Buf(&rb),
            Arg::Buf(&wib),
            Arg::Buf(&swb),
            Arg::Buf(&fault),
            Arg::Bytes(params),
        ],
        Grid::Threads((n * data_cols) as u64),
    )]);
    let (mut wd, mut wwi, mut wwr) = (data_init, wi_init, wr_init);
    for t in 0..n * data_cols {
        let (i, k) = (t / data_cols, t % data_cols);
        let c = rows_cycles[i] as usize;
        wd[k * total + c] = rows[t];
        if k == 0 {
            wwi[c] = wom_idx[i];
        }
        if k < K_MAX_WOM_ROWS_PER_CYCLE {
            let (dst, src) = (
                (c * K_MAX_WOM_ROWS_PER_CYCLE + k) * 5,
                (i * K_MAX_WOM_ROWS_PER_CYCLE + k) * 5,
            );
            wwr[dst..dst + 5].copy_from_slice(&src_wom[src..src + 5]);
        }
    }
    assert!(data.read() == wd);
    assert!(wi.read() == wwi);
    assert!(wr.read() == wwr);
    assert_eq!(fault.read(), vec![0]);
}

// ------------------------------------------------------------------------------------------------
// extern.h handlers, through a probe kernel compiled at run time from the shipped headers.

/// The probe source: fp.h, fpext.h, cuda2metal.h, context.h, extern.h inlined in include order
/// (MTLCompileOptions has no include path), then three probe kernels.
fn extern_probe_source() -> String {
    let shared = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../build_kernel/kernels/metal"
    );
    let local = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/metal");
    let mut src = String::new();
    for path in [
        format!("{shared}/fp.h"),
        format!("{shared}/fpext.h"),
        format!("{shared}/cuda2metal.h"),
        format!("{local}/context.h"),
        format!("{local}/extern.h"),
    ] {
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        for line in text.lines() {
            let t = line.trim_start();
            if t.starts_with("#include \"") || t.starts_with("#pragma once") {
                continue;
            }
            src.push_str(line);
            src.push('\n');
        }
    }
    src.push_str(EXTERN_PROBE_KERNELS);
    src
}

const EXTERN_PROBE_KERNELS: &str = r#"
using namespace metal;
// Per cycle: womRead(addrIn[c]), readIOPBody, plonkRead_wom; out[c*13 + ..] = 4 + 4 + 5 words.
kernel void probe_read(const device FpExt* wom [[buffer(0)]], device PreflightCycle* cycles [[buffer(1)]],
                       const device FpExt* iops [[buffer(2)]], device WomArgumentRow* womRows [[buffer(3)]],
                       device uint32_t* womIndex [[buffer(4)]], const device uint32_t* addrIn [[buffer(5)]],
                       device Fp* out [[buffer(6)]], device uint32_t* badOut [[buffer(7)]],
                       device atomic_uint* fault [[buffer(8)]], constant RecExecParams& p [[buffer(9)]],
                       uint cycle [[thread_position_in_grid]]) {
  if (cycle >= p.numCycles) return;
  ExecCtx c = make_exec_ctx(nullptr, nullptr, nullptr, wom, cycles, iops, womRows, womIndex, p);
  thread ExecCtx* ctx = &c;
  Fp extern_args[96];
  Fp extern_outs[32];
  extern_args[0] = Fp(addrIn[cycle]);
  extern_womRead(ctx, cycle, "wom", extern_args, extern_outs);
  for (uint k = 0; k < 4; k++) out[cycle * 13 + k] = extern_outs[k];
  extern_readIOPBody(ctx, cycle, "", extern_args, extern_outs);
  for (uint k = 0; k < 4; k++) out[cycle * 13 + 4 + k] = extern_outs[k];
  extern_plonkRead_wom(ctx, cycle, "wom", extern_args, extern_outs);
  for (uint k = 0; k < 5; k++) out[cycle * 13 + 8 + k] = extern_outs[k];
  extern_readIOPHeader(ctx, cycle, "", extern_args, extern_outs);
  extern_womWrite(ctx, cycle, "wom", extern_args, extern_outs);
  extern_readCoefficients(ctx, cycle, "", extern_args, extern_outs);
  extern_log(ctx, cycle, "%u", extern_args, extern_outs);
  badOut[cycle] = ctx->bad;
  r0_fault_flush(fault, ctx->bad);
}
// Per cycle: nWrites[c] calls of plonkWrite_wom with args = (row j of argsIn[c]).
kernel void probe_write(device WomArgumentRow* womRows [[buffer(0)]], device uint32_t* womIndex [[buffer(1)]],
                        const device uint32_t* nWrites [[buffer(2)]], const device Fp* argsIn [[buffer(3)]],
                        device uint32_t* badOut [[buffer(4)]], device atomic_uint* fault [[buffer(5)]],
                        constant RecExecParams& p [[buffer(6)]], uint cycle [[thread_position_in_grid]]) {
  if (cycle >= p.numCycles) return;
  ExecCtx c = make_exec_ctx(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, womRows, womIndex, p);
  Fp extern_args[96];
  Fp extern_outs[32];
  for (uint j = 0; j < nWrites[cycle]; j++) {
    for (uint k = 0; k < 5; k++) extern_args[k] = argsIn[(cycle * 10 + j) * 5 + k];
    extern_plonkWrite_wom(&c, cycle, "wom", extern_args, extern_outs);
  }
  badOut[cycle] = c.bad;
  r0_fault_flush(fault, c.bad);
}
// Thread t works on cycle t + shift: WriteAccum(argsIn[t]) then ReadAccum into out[t].
kernel void probe_accum(device FpExt* accum [[buffer(0)]], const device Fp* argsIn [[buffer(1)]],
                        device Fp* out [[buffer(2)]], device uint32_t* badOut [[buffer(3)]],
                        device atomic_uint* fault [[buffer(4)]], constant RecAccumParams& p [[buffer(5)]],
                        constant uint32_t& count [[buffer(6)]], constant uint32_t& shift [[buffer(7)]],
                        uint t [[thread_position_in_grid]]) {
  if (t >= count) return;
  AccumCtx c = make_accum_ctx(nullptr, nullptr, nullptr, nullptr, nullptr, accum, p);
  Fp a[96];
  Fp o[32];
  for (uint k = 0; k < 4; k++) a[k] = argsIn[t * 4 + k];
  extern_plonkWriteAccum_wom(&c, t + shift, "wom", a, o);
  extern_plonkReadAccum_wom(&c, t + shift, "wom", a, o);
  for (uint k = 0; k < 4; k++) out[t * 4 + k] = o[k];
  badOut[t] = c.bad;
  r0_fault_flush(fault, c.bad);
}
"#;

#[test]
fn c1_externs_match_cpu_and_clamp() {
    let gpu = Gpu::from_source(&extern_probe_source());
    let mut rng = Rng(0xe7);
    let zero = raw(Elem::ZERO);

    // --- probe_read: womRead / readIOPBody / plonkRead_wom.
    let (num_cycles, num_woms, num_iops, num_wom_rows) = (64usize, 40usize, 50usize, 30usize);
    let wom: Vec<u32> = (0..num_woms * 4).map(|_| raw(rng.elem())).collect();
    let iops: Vec<u32> = (0..num_iops * 4).map(|_| raw(rng.elem())).collect();
    let wom_rows: Vec<u32> = (0..num_wom_rows * 5).map(|_| raw(rng.elem())).collect();
    // Cycle c: addr = c (out of range from 40), iopIdx = c (out of range from 50), womIndex = c
    // (out of range from 30).
    let addr_in: Vec<u32> = (0..num_cycles as u32).collect();
    let cycles_init: Vec<u32> = (0..num_cycles as u32).flat_map(|c| [c, 1]).collect();
    let wom_index_init: Vec<u32> = (0..num_cycles as u32).collect();
    let params = RecExecParams {
        total_cycles: 64,
        num_cycles: num_cycles as u32,
        num_woms: num_woms as u32,
        num_iops: num_iops as u32,
        num_wom_rows: num_wom_rows as u32,
    };
    let (wb, cb, ib, rb, xb, ab) = (
        gpu.buffer_from(&wom),
        gpu.buffer_from(&cycles_init),
        gpu.buffer_from(&iops),
        gpu.buffer_from(&wom_rows),
        gpu.buffer_from(&wom_index_init),
        gpu.buffer_from(&addr_in),
    );
    let out = gpu.buffer(num_cycles * 13);
    let bad = gpu.buffer(num_cycles);
    let fault = gpu.buffer_from(&[0]);
    gpu.run(&[(
        "probe_read",
        vec![
            Arg::Buf(&wb),
            Arg::Buf(&cb),
            Arg::Buf(&ib),
            Arg::Buf(&rb),
            Arg::Buf(&xb),
            Arg::Buf(&ab),
            Arg::Buf(&out),
            Arg::Buf(&bad),
            Arg::Buf(&fault),
            params.arg(),
        ],
        Grid::Threads(num_cycles as u64),
    )]);
    let (got, got_bad, got_cycles, got_wi) = (out.read(), bad.read(), cb.read(), xb.read());
    for c in 0..num_cycles {
        let o = &got[c * 13..c * 13 + 13];
        let want_wom = if c < num_woms {
            wom[c * 4..c * 4 + 4].to_vec()
        } else {
            vec![zero; 4]
        };
        let want_iop = if c < num_iops {
            iops[c * 4..c * 4 + 4].to_vec()
        } else {
            vec![zero; 4]
        };
        let want_row = if c < num_wom_rows {
            wom_rows[c * 5..c * 5 + 5].to_vec()
        } else {
            vec![zero; 5]
        };
        assert_eq!(&o[0..4], &want_wom[..], "womRead cycle {c}");
        assert_eq!(&o[4..8], &want_iop[..], "readIOPBody cycle {c}");
        assert_eq!(&o[8..13], &want_row[..], "plonkRead_wom cycle {c}");
        let want_bad = (c >= num_woms) || (c >= num_iops) || (c >= num_wom_rows);
        assert_eq!(got_bad[c], want_bad as u32, "bad cycle {c}");
        assert_eq!(got_cycles[c * 2], c as u32 + 1, "iopIdx++ cycle {c}");
        assert_eq!(got_cycles[c * 2 + 1], 1, "isParSafe untouched");
        assert_eq!(got_wi[c], c as u32 + 1, "womIndex++ cycle {c}");
    }
    assert_eq!(fault.read(), vec![1]);

    // --- probe_write: plonkWrite_wom fills slots [c*9, c*9 + n) in order; the 10th write faults.
    let num_cycles = 40usize;
    let n_writes: Vec<u32> = (0..num_cycles)
        .map(|c| if c == 7 { 10 } else { rng.below(10) })
        .collect();
    let args_in: Vec<u32> = (0..num_cycles * 10 * 5).map(|_| raw(rng.elem())).collect();
    let num_wom_rows = num_cycles * K_MAX_WOM_ROWS_PER_CYCLE;
    let rows_init = vec![INVALID; num_wom_rows * 5];
    let params = RecExecParams {
        total_cycles: 64,
        num_cycles: num_cycles as u32,
        num_wom_rows: num_wom_rows as u32,
        ..Default::default()
    };
    let (rb, xb, nb, argb) = (
        gpu.buffer_from(&rows_init),
        gpu.buffer_from(&vec![0; num_cycles]),
        gpu.buffer_from(&n_writes),
        gpu.buffer_from(&args_in),
    );
    let bad = gpu.buffer(num_cycles);
    let fault = gpu.buffer_from(&[0]);
    gpu.run(&[(
        "probe_write",
        vec![
            Arg::Buf(&rb),
            Arg::Buf(&xb),
            Arg::Buf(&nb),
            Arg::Buf(&argb),
            Arg::Buf(&bad),
            Arg::Buf(&fault),
            params.arg(),
        ],
        Grid::Threads(num_cycles as u64),
    )]);
    let mut want_rows = rows_init.clone();
    for c in 0..num_cycles {
        for j in 0..(n_writes[c] as usize).min(K_MAX_WOM_ROWS_PER_CYCLE) {
            let dst = (c * K_MAX_WOM_ROWS_PER_CYCLE + j) * 5;
            let src = (c * 10 + j) * 5;
            want_rows[dst..dst + 5].copy_from_slice(&args_in[src..src + 5]);
        }
    }
    assert!(rb.read() == want_rows, "plonkWrite_wom rows");
    assert!(xb.read() == n_writes, "womIndex counts every call");
    let got_bad = bad.read();
    for c in 0..num_cycles {
        assert_eq!(got_bad[c], (n_writes[c] > 9) as u32, "bad cycle {c}");
    }
    assert_eq!(fault.read(), vec![1]);

    // --- probe_accum: write/read round trip; cycles >= workCycles fault and read zero.
    let work = 100u32;
    let accum = gpu.buffer_from(&vec![0x77u32; work as usize * 4 + 4]);
    let args_in: Vec<u32> = (0..work as usize * 4).map(|_| raw(rng.elem())).collect();
    let argb = gpu.buffer_from(&args_in);
    for shift in [0u32, 10] {
        let out = gpu.buffer(work as usize * 4);
        let bad = gpu.buffer(work as usize);
        let fault = gpu.buffer_from(&[0]);
        gpu.run(&[(
            "probe_accum",
            vec![
                Arg::Buf(&accum),
                Arg::Buf(&argb),
                Arg::Buf(&out),
                Arg::Buf(&bad),
                Arg::Buf(&fault),
                Arg::Bytes(words_to_bytes(&[64, work, 0, 0])),
                Arg::U32(work),
                Arg::U32(shift),
            ],
            Grid::Threads(work as u64),
        )]);
        let (got, got_bad) = (out.read(), bad.read());
        for t in 0..work as usize {
            let in_range = t as u32 + shift < work;
            let want = if in_range {
                args_in[t * 4..t * 4 + 4].to_vec()
            } else {
                vec![zero; 4]
            };
            assert_eq!(&got[t * 4..t * 4 + 4], &want[..], "shift {shift} t {t}");
            assert_eq!(got_bad[t], !in_range as u32, "shift {shift} bad {t}");
        }
        assert_eq!(fault.read(), vec![(shift != 0) as u32]);
        assert_eq!(
            &accum.read()[work as usize * 4..],
            &[0x77; 4],
            "guard past workCycles"
        );
    }
}
