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

// Day-0 Metal runtime probes (plan item D0, docs/metal-port/PLAN.md section 3 and errata 8.5).
//
// Every probe is a subcommand. `all` runs each case in a child process (so a crash or a GPU
// fault in one case is reported instead of taking down the rest) and prints one
// `RESULT <case>: <verdict> | <details>` line per case. Probe .metal sources live one directory
// up and are compiled with the offline toolchain (`xcrun metal -std=metal3.2 -O2`, then
// `xcrun metallib`), i.e. the same compiler the risc0 build uses, not the runtime compiler.
//
//   source /Users/user/zkminer/env.sh
//   cargo run --release --manifest-path risc0/build_kernel/kernels/metal/probes/runner/Cargo.toml -- all
//   cargo run --release --manifest-path ... -- watchdog 60

// objc 0.2 msg_send! expands to cfg(feature = "cargo-clippy").
#![allow(unexpected_cfgs)]

use std::{
    ffi::{c_void, CStr},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use metal::{
    foreign_types::ForeignType, Buffer, BufferRef, CommandBufferRef, CommandQueue,
    ComputeCommandEncoderRef, ComputePassDescriptor, ComputePipelineState,
    CounterSampleBufferDescriptor, Device, Library, MTLCommandBufferStatus,
    MTLCounterSamplingPoint, MTLGPUFamily, MTLResourceOptions, MTLResourceUsage, MTLSize,
    MTLStorageMode, NSRange,
};
use objc::{msg_send, rc::autoreleasepool, runtime::Object, sel, sel_impl};

const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

// ---------------------------------------------------------------------------------------------
// helpers

fn probes_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn air_dir() -> PathBuf {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/probe-air");
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Compile `<stem>.metal` files to .air and link them into one metallib; returns the bytes or
/// the linker's stderr.
fn build_metallib(stems: &[&str]) -> Result<Vec<u8>, String> {
    let out = air_dir();
    let mut airs = vec![];
    for stem in stems {
        let src = probes_dir().join(format!("{stem}.metal"));
        let air = out.join(format!("{stem}.air"));
        let o = Command::new("xcrun")
            .args(["metal", "-std=metal3.2", "-O2", "-c"])
            .arg(&src)
            .arg("-o")
            .arg(&air)
            .output()
            .expect("xcrun not found: source /Users/user/zkminer/env.sh");
        if !o.status.success() {
            return Err(format!(
                "compile {stem}: {}",
                String::from_utf8_lossy(&o.stderr)
            ));
        }
        airs.push(air);
    }
    let lib = out.join(format!("{}.metallib", stems.join("+")));
    let o = Command::new("xcrun")
        .arg("metallib")
        .args(&airs)
        .arg("-o")
        .arg(&lib)
        .output()
        .unwrap();
    if !o.status.success() {
        return Err(format!(
            "metallib: {}",
            String::from_utf8_lossy(&o.stderr).trim()
        ));
    }
    Ok(std::fs::read(lib).unwrap())
}

fn library(dev: &Device, stems: &[&str]) -> Library {
    let bytes = build_metallib(stems).unwrap_or_else(|e| panic!("{e}"));
    dev.new_library_with_data(&bytes)
        .expect("new_library_with_data")
}

fn pso(dev: &Device, lib: &Library, name: &str) -> Result<ComputePipelineState, String> {
    let f = lib.get_function(name, None)?;
    dev.new_compute_pipeline_state_with_function(&f)
}

fn obj<T>(r: &T) -> *mut Object {
    r as *const T as *mut Object
}

fn nsstring(s: *mut Object) -> String {
    if s.is_null() {
        return String::new();
    }
    unsafe {
        let c: *const std::os::raw::c_char = msg_send![s, UTF8String];
        CStr::from_ptr(c).to_string_lossy().into_owned()
    }
}

/// `-[MTLCommandBuffer error]` (metal-rs 0.29 has no accessor), as "code N: description".
fn cb_error(cb: &CommandBufferRef) -> String {
    unsafe {
        let err: *mut Object = msg_send![obj(cb), error];
        if err.is_null() {
            return "no error".into();
        }
        let code: i64 = msg_send![err, code];
        let desc: *mut Object = msg_send![err, localizedDescription];
        format!("code {code}: {}", nsstring(desc))
    }
}

/// GPUEndTime - GPUStartTime in seconds.
fn gpu_seconds(cb: &CommandBufferRef) -> f64 {
    unsafe {
        let s: f64 = msg_send![obj(cb), GPUStartTime];
        let e: f64 = msg_send![obj(cb), GPUEndTime];
        e - s
    }
}

struct CbResult {
    status: MTLCommandBufferStatus,
    error: String,
    gpu_s: f64,
    wall_s: f64,
}

impl CbResult {
    fn ok(&self) -> bool {
        self.status == MTLCommandBufferStatus::Completed
    }
    fn describe(&self) -> String {
        format!(
            "status={:?} ({}) gpu={:.3} ms wall={:.3} ms",
            self.status,
            self.error,
            self.gpu_s * 1e3,
            self.wall_s * 1e3
        )
    }
}

/// Encode with `f` into one compute encoder, commit, wait.
fn run(q: &CommandQueue, f: impl FnOnce(&ComputeCommandEncoderRef)) -> CbResult {
    autoreleasepool(|| {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let t = Instant::now();
        f(enc);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let wall_s = t.elapsed().as_secs_f64();
        CbResult {
            status: cb.status(),
            error: cb_error(cb),
            gpu_s: gpu_seconds(cb),
            wall_s,
        }
    })
}

fn sz(x: u64) -> MTLSize {
    MTLSize::new(x, 1, 1)
}

fn buf_u32(dev: &Device, n: usize, fill: u32) -> Buffer {
    let b = dev.new_buffer((n * 4) as u64, SHARED);
    slice_mut::<u32>(&b, n).fill(fill);
    b
}

fn slice_mut<T>(b: &BufferRef, n: usize) -> &mut [T] {
    assert!(n * std::mem::size_of::<T>() <= b.length() as usize);
    unsafe { std::slice::from_raw_parts_mut(b.contents() as *mut T, n) }
}

fn set_val<T>(enc: &ComputeCommandEncoderRef, idx: u64, v: &T) {
    enc.set_bytes(
        idx,
        std::mem::size_of::<T>() as u64,
        v as *const T as *const c_void,
    );
}

fn result(case: &str, verdict: &str, details: impl AsRef<str>) {
    println!("RESULT {case}: {verdict} | {}", details.as_ref());
}

fn setup() -> (Device, CommandQueue) {
    let dev = Device::system_default().expect("no Metal device");
    let q = dev.new_command_queue();
    (dev, q)
}

// ---------------------------------------------------------------------------------------------
// probes

fn probe_device() {
    let (dev, _q) = setup();
    let mut fams = vec![];
    let known: &[(&str, MTLGPUFamily)] = &[
        ("Apple7", MTLGPUFamily::Apple7),
        ("Apple8", MTLGPUFamily::Apple8),
        ("Apple9", MTLGPUFamily::Apple9),
        ("Mac2", MTLGPUFamily::Mac2),
        ("Common3", MTLGPUFamily::Common3),
        ("Metal3", MTLGPUFamily::Metal3),
    ];
    for (n, f) in known {
        fams.push(format!("{n}={}", dev.supports_family(*f)));
    }
    // Values metal-rs 0.29 does not know: MTLGPUFamilyApple10 = 1010, MTLGPUFamilyMetal4 = 5002.
    for (n, raw) in [
        ("Apple10(raw 1010)", 1010i64),
        ("Apple11(raw 1011)", 1011),
        ("Metal4(raw 5002)", 5002),
    ] {
        let s: bool = unsafe {
            let r: objc::runtime::BOOL = msg_send![obj(dev.as_ref()), supportsFamily: raw];
            r == objc::runtime::YES
        };
        fams.push(format!("{n}={s}"));
    }
    let arch = unsafe {
        let a: *mut Object = msg_send![obj(dev.as_ref()), architecture];
        if a.is_null() {
            "?".into()
        } else {
            nsstring(msg_send![a, name])
        }
    };
    result(
        "device",
        "VALUE",
        format!(
            "name={} arch={} unified={} max_tg_mem={} max_threads_per_tg={:?} recommended_ws={} GiB max_buffer_length={} GiB",
            dev.name(),
            arch,
            dev.has_unified_memory(),
            dev.max_threadgroup_memory_length(),
            (dev.max_threads_per_threadgroup().width, dev.max_threads_per_threadgroup().height, dev.max_threads_per_threadgroup().depth),
            dev.recommended_max_working_set_size() as f64 / (1u64 << 30) as f64,
            dev.max_buffer_length() as f64 / (1u64 << 30) as f64,
        ),
    );
    result("gpu-family", "VALUE", fams.join(" "));
}

fn probe_host_name() {
    let (dev, q) = setup();
    let lib = library(&dev, &["host_name"]);
    let mut names = lib.function_names();
    names.sort();
    let n = 64usize;
    let mut ok = true;
    let mut details = vec![format!("functions={names:?}")];
    for (name, r, c) in [("ct_x_8_coalesced", 8u32, 1u32), ("ct_x_1_plain", 1, 0)] {
        match pso(&dev, &lib, name) {
            Err(e) => {
                ok = false;
                details.push(format!("{name}: lookup/PSO failed: {e}"));
            }
            Ok(p) => {
                let out = buf_u32(&dev, n, 0xdead);
                let res = run(&q, |enc| {
                    enc.set_compute_pipeline_state(&p);
                    enc.set_buffer(0, Some(&out), 0);
                    enc.dispatch_threads(sz(n as u64), sz(32));
                });
                let got = slice_mut::<u32>(&out, n);
                let good = res.ok() && got.iter().enumerate().all(|(g, &v)| v == g as u32 * r + c);
                ok &= good;
                details.push(format!("{name}: correct={good}"));
            }
        }
    }
    let base = lib.get_function("ct_x", None).is_ok();
    details.push(format!("unmangled template name 'ct_x' resolvable={base}"));
    result(
        "host-name",
        if ok { "ACCEPT" } else { "REJECT" },
        details.join("; "),
    );
}

fn probe_extern() {
    let (dev, q) = setup();
    let only_use = build_metallib(&["extern_use"]);
    let lib = library(&dev, &["extern_use", "extern_def"]);
    let p = pso(&dev, &lib, "extern_call");
    let mut details = vec![format!(
        "metallib(extern_use.air alone): {}",
        match &only_use {
            Ok(_) => "LINKED (unexpected)".to_string(),
            Err(e) => format!(
                "link error `{}`",
                e.lines().find(|l| l.contains("rror")).unwrap_or(e)
            ),
        }
    )];
    let ok = match p {
        Err(e) => {
            details.push(format!("PSO failed: {e}"));
            false
        }
        Ok(p) => {
            let n = 256;
            let out = buf_u32(&dev, n, 0);
            let res = run(&q, |enc| {
                enc.set_compute_pipeline_state(&p);
                enc.set_buffer(0, Some(&out), 0);
                enc.dispatch_threads(sz(n as u64), sz(64));
            });
            let good = res.ok()
                && slice_mut::<u32>(&out, n)
                    .iter()
                    .enumerate()
                    .all(|(g, &v)| v == g as u32 * 5 + 1);
            details.push(format!("metallib(use+def) runtime correct={good}"));
            good
        }
    };
    result(
        "extern",
        if ok && only_use.is_err() {
            "ACCEPT"
        } else {
            "REJECT"
        },
        details.join("; "),
    );
}

fn probe_designated() {
    let (dev, q) = setup();
    let lib = library(&dev, &["designated_init"]);
    let p = pso(&dev, &lib, "designated").unwrap();
    let out = buf_u32(&dev, 5, 0);
    let res = run(&q, |enc| {
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&out), 0);
        enc.dispatch_threads(sz(5), sz(5));
    });
    let got = slice_mut::<u32>(&out, 5).to_vec();
    let want = vec![1023u32, 4056, 7009, 10122, 107];
    result(
        "designated",
        if res.ok() && got == want {
            "ACCEPT"
        } else {
            "REJECT"
        },
        format!("got={got:?} want={want:?} (omitted .b zero-initialised)"),
    );
}

#[repr(C)]
struct Jobs {
    ptrs: [u64; 4],
    n: u32,
    _pad: u32,
}

fn probe_gpu_address(use_resource: bool) {
    let case = if use_resource {
        "gpu-address+use_resource"
    } else {
        "gpu-address-no-use_resource"
    };
    let (dev, q) = setup();
    let lib = library(&dev, &["gpu_address"]);
    let p = pso(&dev, &lib, "gpu_address").unwrap();
    let n = 4096usize;
    let bufs: Vec<Buffer> = (0..4).map(|_| buf_u32(&dev, n, 0)).collect();
    let jobs = Jobs {
        ptrs: [
            bufs[0].gpu_address(),
            bufs[1].gpu_address(),
            bufs[2].gpu_address(),
            bufs[3].gpu_address(),
        ],
        n: n as u32,
        _pad: 0,
    };
    let res = run(&q, |enc| {
        enc.set_compute_pipeline_state(&p);
        set_val(enc, 0, &jobs);
        if use_resource {
            for b in &bufs {
                enc.use_resource(b, MTLResourceUsage::Write);
            }
        }
        enc.dispatch_threads(sz(4 * n as u64), sz(256));
    });
    let mut bad = 0;
    for (k, b) in bufs.iter().enumerate() {
        for (i, &v) in slice_mut::<u32>(b, n).iter().enumerate() {
            if v != (i * 4 + k) as u32 + 1 {
                bad += 1;
            }
        }
    }
    result(
        case,
        if res.ok() && bad == 0 {
            "ACCEPT"
        } else {
            "REJECT"
        },
        format!(
            "sizeof(Jobs)={} addrs={:x?} mismatches={bad}/{} {}",
            std::mem::size_of::<Jobs>(),
            jobs.ptrs,
            4 * n,
            res.describe()
        ),
    );
}

fn probe_tg_static(which: &str) {
    let (dev, q) = setup();
    let stem = format!("tg_static_{which}");
    let lib = library(&dev, &[&stem]);
    let case = format!("tg-static-{which}");
    match pso(&dev, &lib, &stem) {
        Err(e) => result(&case, "REJECT", format!("PSO creation failed: {e}")),
        Ok(p) => {
            let out = buf_u32(&dev, 256, 0xdead);
            let res = run(&q, |enc| {
                enc.set_compute_pipeline_state(&p);
                enc.set_buffer(0, Some(&out), 0);
                enc.dispatch_thread_groups(
                    sz(1),
                    sz(256.min(p.max_total_threads_per_threadgroup())),
                );
            });
            let bad: u32 = slice_mut::<u32>(&out, 256).iter().sum();
            result(
                &case,
                if res.ok() && bad == 0 { "ACCEPT" } else { "REJECT" },
                format!(
                    "PSO created: static_tg_mem={} max_threads={} ; dispatch {} mismatch_sum={bad:#x}",
                    p.static_threadgroup_memory_length(),
                    p.max_total_threads_per_threadgroup(),
                    res.describe()
                ),
            );
        }
    }
}

fn probe_tg_dynamic(bytes: u64) {
    let (dev, q) = setup();
    let lib = library(&dev, &["tg_dynamic"]);
    let p = pso(&dev, &lib, "tg_dynamic").unwrap();
    let out = buf_u32(&dev, 256, 0xdead);
    let words = (bytes / 4) as u32;
    let res = run(&q, |enc| {
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&out), 0);
        set_val(enc, 1, &words);
        enc.set_threadgroup_memory_length(0, bytes);
        enc.dispatch_thread_groups(sz(4), sz(256));
    });
    let bad: u64 = slice_mut::<u32>(&out, 256).iter().map(|&x| x as u64).sum();
    result(
        &format!("tg-dynamic-{bytes}"),
        match (
            res.ok() && bad == 0,
            bytes > dev.max_threadgroup_memory_length(),
        ) {
            (true, false) => "ACCEPT",
            (true, true) => "RUNS ABOVE LIMIT (no error reported; validation layer aborts)",
            (false, _) => "REJECT (silent corruption, status Completed)",
        },
        format!(
            "device max_tg_mem={} mismatch_sum={bad:#x} {}",
            dev.max_threadgroup_memory_length(),
            res.describe()
        ),
    );
}

fn probe_counters() {
    let (dev, q) = setup();
    let mut sets = vec![];
    for s in dev.counter_sets() {
        let counters: Vec<String> = unsafe {
            let arr: *mut Object = msg_send![obj(s.as_ref()), counters];
            let count: u64 = msg_send![arr, count];
            (0..count)
                .map(|i| {
                    let c: *mut Object = msg_send![arr, objectAtIndex: i];
                    nsstring(msg_send![c, name])
                })
                .collect()
        };
        sets.push(format!("{}{counters:?}", s.name()));
    }
    let points = [
        ("stage", MTLCounterSamplingPoint::AtStageBoundary),
        ("draw", MTLCounterSamplingPoint::AtDrawBoundary),
        ("dispatch", MTLCounterSamplingPoint::AtDispatchBoundary),
        ("tile", MTLCounterSamplingPoint::AtTileDispatchBoundary),
        ("blit", MTLCounterSamplingPoint::AtBlitBoundary),
    ];
    let support: Vec<String> = points
        .iter()
        .map(|(n, p)| format!("{n}={}", dev.supports_counter_sampling(*p)))
        .collect();
    result(
        "counter-sets",
        "VALUE",
        format!("sets=[{}] sampling: {}", sets.join(", "), support.join(" ")),
    );

    // Try a timestamp sample buffer at stage (encoder) boundary, and at dispatch boundary when
    // supported.
    let ts = dev
        .counter_sets()
        .into_iter()
        .find(|s| s.name() == "timestamp");
    let Some(ts) = ts else {
        result("counter-sampling", "REJECT", "no `timestamp` counter set");
        return;
    };
    let desc = CounterSampleBufferDescriptor::new();
    desc.set_counter_set(&ts);
    desc.set_sample_count(4);
    desc.set_storage_mode(MTLStorageMode::Shared);
    let sb = match dev.new_counter_sample_buffer_with_descriptor(&desc) {
        Ok(sb) => sb,
        Err(e) => {
            result(
                "counter-sampling",
                "REJECT",
                format!("newCounterSampleBuffer failed: {e}"),
            );
            return;
        }
    };
    let lib = library(&dev, &["dispatch"]);
    let p = pso(&dev, &lib, "inc_kernel").unwrap();
    let n = 1u32 << 22;
    let data = buf_u32(&dev, n as usize, 1);
    let resolved = dev.new_buffer(4 * 8, SHARED);
    let dispatch_ok = dev.supports_counter_sampling(MTLCounterSamplingPoint::AtDispatchBoundary);
    let stage_ok = dev.supports_counter_sampling(MTLCounterSamplingPoint::AtStageBoundary);
    let (mut c0, mut g0, mut c1, mut g1) = (0u64, 0u64, 0u64, 0u64);
    dev.sample_timestamps(&mut c0, &mut g0);
    let (status, gpu_s) = autoreleasepool(|| {
        let cb = q.new_command_buffer();
        let enc = if stage_ok {
            let pd = ComputePassDescriptor::new();
            let att = pd.sample_buffer_attachments().object_at(0).unwrap();
            att.set_sample_buffer(&sb);
            att.set_start_of_encoder_sample_index(0);
            att.set_end_of_encoder_sample_index(1);
            cb.compute_command_encoder_with_descriptor(pd)
        } else {
            cb.new_compute_command_encoder()
        };
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&data), 0);
        set_val(enc, 1, &n);
        if dispatch_ok {
            enc.sample_counters_in_buffer(&sb, 2, true);
        }
        enc.dispatch_threads(sz(n as u64), sz(256));
        if dispatch_ok {
            enc.sample_counters_in_buffer(&sb, 3, true);
        }
        enc.end_encoding();
        let blit = cb.new_blit_command_encoder();
        blit.resolve_counters(&sb, NSRange::new(0, 4), &resolved, 0);
        blit.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        ((cb.status(), cb_error(cb)), gpu_seconds(cb))
    });
    std::thread::sleep(Duration::from_millis(20));
    dev.sample_timestamps(&mut c1, &mut g1);
    let blit_r = slice_mut::<u64>(&resolved, 4).to_vec();
    // CPU-side resolve: -[MTLCounterSampleBuffer resolveCounterRange:] -> NSData (not wrapped by
    // metal-rs 0.29).
    let cpu_r: Vec<u64> = unsafe {
        let data: *mut Object =
            msg_send![obj(sb.as_ref()), resolveCounterRange: NSRange::new(0, 4)];
        if data.is_null() {
            vec![]
        } else {
            let len: u64 = msg_send![data, length];
            let bytes: *const u64 = msg_send![data, bytes];
            std::slice::from_raw_parts(bytes, (len / 8) as usize).to_vec()
        }
    };
    let r = if cpu_r.len() == 4 {
        cpu_r.clone()
    } else {
        blit_r.clone()
    };
    let tick_ratio = (c1 - c0) as f64 / (g1 - g0).max(1) as f64;
    result(
        "counter-sampling",
        if status.0 == MTLCommandBufferStatus::Completed && (stage_ok || dispatch_ok) { "VALUE" } else { "REJECT" },
        format!(
            "stage_boundary={stage_ok} dispatch_boundary={dispatch_ok} cpu_resolve={cpu_r:?} blit_resolve={blit_r:?} (unsampled slots resolve to 0; GPU ticks are ns when cpu/gpu tick ratio = 1) \
             stage_delta_ticks={} dispatch_delta_ticks={} cpu/gpu tick ratio={tick_ratio:.4} cmdbuf GPU time={:.3} ms status={:?} ({})",
            r[1].wrapping_sub(r[0]),
            r[3].wrapping_sub(r[2]),
            gpu_s * 1e3,
            status.0,
            status.1
        ),
    );
}

fn probe_watchdog(seconds: f64) {
    let (dev, q) = setup();
    let lib = library(&dev, &["watchdog"]);
    let p = pso(&dev, &lib, "spin").unwrap();
    let out = buf_u32(&dev, 1, 0);
    let go = |iters: u64| {
        let v = [(iters & 0xffff_ffff) as u32, (iters >> 32) as u32];
        run(&q, |enc| {
            enc.set_compute_pipeline_state(&p);
            enc.set_buffer(0, Some(&out), 0);
            set_val(enc, 1, &v);
            enc.dispatch_threads(sz(1), sz(1));
        })
    };
    let cal_iters = 1u64 << 26;
    let cal = go(cal_iters);
    let per_iter = cal.gpu_s / cal_iters as f64;
    let iters = (seconds / per_iter) as u64;
    eprintln!(
        "calibration: {} ; {:.3} ns/iter -> {iters} iters for {seconds} s",
        cal.describe(),
        per_iter * 1e9
    );
    let res = go(iters);
    let after = go(1 << 20);
    result(
        &format!("watchdog-{seconds}s"),
        if res.ok() { "NOT KILLED" } else { "KILLED" },
        format!(
            "calibration {:.3} ns/iter; long dispatch ({iters} iters): {} ; follow-up small dispatch: {}",
            per_iter * 1e9,
            res.describe(),
            after.describe()
        ),
    );
}

fn probe_assert() {
    let (dev, q) = setup();
    let lib = library(&dev, &["assert_release"]);
    let p = pso(&dev, &lib, "assert_release").unwrap();
    let n = 128;
    let out = buf_u32(&dev, n, 0);
    let res = run(&q, |enc| {
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&out), 0);
        enc.dispatch_threads(sz(n as u64), sz(64));
    });
    let good = slice_mut::<u32>(&out, n)
        .iter()
        .enumerate()
        .all(|(g, &v)| v == g as u32 + 7);
    result(
        "assert",
        if res.ok() && good { "ACCEPT" } else { "REJECT" },
        format!(
            "always-false assert(): stores after it executed={good} {} (no-op, matches ((void)0))",
            res.describe()
        ),
    );
}

fn probe_maxthreads() {
    let (dev, q) = setup();
    let lib = library(&dev, &["maxthreads_768"]);
    let p = pso(&dev, &lib, "k768").unwrap();
    let pd = pso(&dev, &lib, "kdefault").unwrap();
    let n = 768 * 4;
    let out = buf_u32(&dev, n, 0);
    let res = run(&q, |enc| {
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&out), 0);
        enc.dispatch_thread_groups(sz(4), sz(768));
    });
    let good = slice_mut::<u32>(&out, n)
        .iter()
        .enumerate()
        .all(|(g, &v)| v == g as u32);
    result(
        "maxthreads-768",
        if p.max_total_threads_per_threadgroup() == 768 && res.ok() && good { "ACCEPT" } else { "VALUE" },
        format!(
            "k768 PSO maxTotalThreadsPerThreadgroup={} threadExecutionWidth={} ; default kernel={} ; 768-thread groups correct={good} {}",
            p.max_total_threads_per_threadgroup(),
            p.thread_execution_width(),
            pd.max_total_threads_per_threadgroup(),
            res.describe()
        ),
    );
}

fn probe_mul_bench() {
    let (dev, q) = setup();
    let lib = library(&dev, &["mul_bench"]);
    let threads: u64 = 1 << 20;
    let iters: u32 = 2048;
    let out = dev.new_buffer(threads * 8, SHARED);
    let mut lines = vec![];
    for name in ["mul32_lo", "mul32_hi", "mul32_wide", "mul64_lo", "mul64_hi"] {
        let p = pso(&dev, &lib, name).unwrap();
        let mut best = f64::MAX;
        for _ in 0..3 {
            let res = run(&q, |enc| {
                enc.set_compute_pipeline_state(&p);
                enc.set_buffer(0, Some(&out), 0);
                set_val(enc, 1, &iters);
                enc.dispatch_threads(sz(threads), sz(256));
            });
            assert!(res.ok(), "{name}: {}", res.describe());
            best = best.min(res.gpu_s);
        }
        let ops = threads as f64 * iters as f64 * 4.0;
        lines.push(format!(
            "{name}={:.1} Gop/s ({:.2} ms)",
            ops / best / 1e9,
            best * 1e3
        ));
    }
    result(
        "mul-bench",
        "VALUE",
        format!(
            "{} threads x {iters} iters x 4 chains, best of 3, GPU time: {}",
            threads,
            lines.join(", ")
        ),
    );
}

fn probe_simd_shuffle() {
    let (dev, q) = setup();
    let lib = library(&dev, &["simd_shuffle"]);
    let p = pso(&dev, &lib, "simd_shuffle_probe").unwrap();
    let delta = 4u32;
    let mut parts = vec![];
    for tg in [32u64, 24] {
        let out = buf_u32(&dev, 32 * 4, 0xdead_beef);
        let res = run(&q, |enc| {
            enc.set_compute_pipeline_state(&p);
            enc.set_buffer(0, Some(&out), 0);
            set_val(enc, 1, &delta);
            enc.dispatch_thread_groups(sz(1), sz(tg));
        });
        let o = slice_mut::<u32>(&out, 32 * 4);
        let col = |lanes: std::ops::Range<usize>, k: usize| -> Vec<u32> {
            lanes.map(|l| o[l * 4 + k]).collect()
        };
        let t = tg as usize;
        parts.push(format!(
            "tg={tg}: down(delta=4) lanes {}..{} -> {:?}; up(delta=4) lanes 0..4 -> {:?}; shuffle(lane+40) lanes 0..4 -> {:?}; xor16 lanes {}..{} -> {:?} [{}]",
            t - 4,
            t,
            col(t - 4..t, 0),
            col(0..4, 1),
            col(0..4, 2),
            t - 4,
            t,
            col(t - 4..t, 3),
            res.describe()
        ));
    }
    result(
        "simd-shuffle",
        "VALUE",
        format!("own value = 1000+lane. {}", parts.join(" || ")),
    );
}

fn probe_empty_dispatch() {
    let (dev, q) = setup();
    let lib = library(&dev, &["dispatch"]);
    let p = pso(&dev, &lib, "empty_kernel").unwrap();
    let n = 1000;
    // warm-up
    run(&q, |enc| {
        enc.set_compute_pipeline_state(&p);
        enc.dispatch_threads(sz(1), sz(1));
    });
    // A: n dispatches in one command buffer / one encoder.
    let a = run(&q, |enc| {
        enc.set_compute_pipeline_state(&p);
        for _ in 0..n {
            enc.dispatch_thread_groups(sz(1), sz(1));
        }
    });
    // B: one command buffer per dispatch, commit + wait each.
    let t = Instant::now();
    for _ in 0..n {
        run(&q, |enc| {
            enc.set_compute_pipeline_state(&p);
            enc.dispatch_thread_groups(sz(1), sz(1));
        });
    }
    let b = t.elapsed().as_secs_f64();
    // C: one command buffer per dispatch, all committed, wait on the last.
    let t = Instant::now();
    autoreleasepool(|| {
        let mut last = None;
        for _ in 0..n {
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&p);
            enc.dispatch_thread_groups(sz(1), sz(1));
            enc.end_encoding();
            cb.commit();
            last = Some(cb.to_owned());
        }
        last.unwrap().wait_until_completed();
    });
    let c = t.elapsed().as_secs_f64();
    result(
        "empty-dispatch",
        "VALUE",
        format!(
            "N={n}: one cmdbuf: {:.2} us/dispatch wall ({:.2} us/dispatch GPU); cmdbuf per dispatch, commit+wait each: {:.2} us/dispatch; cmdbuf per dispatch, pipelined: {:.2} us/dispatch",
            a.wall_s * 1e6 / n as f64,
            a.gpu_s * 1e6 / n as f64,
            b * 1e6 / n as f64,
            c * 1e6 / n as f64
        ),
    );
}

fn probe_max_dispatches(n: u64) {
    let (dev, q) = setup();
    let lib = library(&dev, &["dispatch"]);
    let p = pso(&dev, &lib, "count_kernel").unwrap();
    let counter = buf_u32(&dev, 1, 0);
    let res = run(&q, |enc| {
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(&counter), 0);
        for _ in 0..n {
            enc.dispatch_thread_groups(sz(1), sz(1));
        }
    });
    let got = slice_mut::<u32>(&counter, 1)[0] as u64;
    result(
        &format!("max-dispatches-{n}"),
        if res.ok() && got == n {
            "ACCEPT"
        } else {
            "REJECT"
        },
        format!(
            "counter={got} {} ({:.2} us/dispatch wall)",
            res.describe(),
            res.wall_s * 1e6 / n as f64
        ),
    );
}

fn run_inc(dev: &Device, q: &CommandQueue, buf: &BufferRef, n: u32) -> CbResult {
    let lib = library(dev, &["dispatch"]);
    let p = pso(dev, &lib, "inc_kernel").unwrap();
    run(q, |enc| {
        enc.set_compute_pipeline_state(&p);
        enc.set_buffer(0, Some(buf), 0);
        set_val(enc, 1, &n);
        enc.dispatch_threads(sz(n as u64), sz(256));
    })
}

/// Raw newBufferWithBytesNoCopy so a nil return is observable (metal-rs would wrap nil).
fn no_copy_raw(dev: &Device, ptr: *mut u8, len: usize) -> Option<Buffer> {
    unsafe {
        let null: *const c_void = std::ptr::null();
        let b: *mut Object = msg_send![obj(dev.as_ref()),
            newBufferWithBytesNoCopy: ptr as *const c_void
            length: len as u64
            options: SHARED
            deallocator: null];
        if b.is_null() {
            None
        } else {
            Some(Buffer::from_ptr(b as *mut metal::MTLBuffer))
        }
    }
}

fn no_copy_case(dev: &Device, q: &CommandQueue, ptr: *mut u8, len: usize) -> String {
    let n = (len / 4) as u32;
    let host = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u32, n as usize) };
    for (i, v) in host.iter_mut().enumerate() {
        *v = i as u32;
    }
    match no_copy_raw(dev, ptr, len) {
        None => "newBufferWithBytesNoCopy returned nil".into(),
        Some(b) => {
            let same = b.contents() as *mut u8 == ptr;
            let res = run_inc(dev, q, &b, n);
            let good = host.iter().enumerate().all(|(i, &v)| v == 2 * i as u32 + 1);
            drop(b);
            format!("buffer created, contents()==host ptr: {same}, GPU writes visible through host ptr: {good}, {}", res.describe())
        }
    }
}

fn probe_no_copy(which: &str) {
    let (dev, q) = setup();
    let page = 16384usize;
    let case = format!("no-copy-{which}");
    match which {
        "aligned" | "unaligned" | "unaligned64" | "oddlen" => {
            let len = 64 * 1024;
            let layout = std::alloc::Layout::from_size_align(len + page, page).unwrap();
            let base = unsafe { std::alloc::alloc_zeroed(layout) };
            let (off, l) = match which {
                "aligned" => (0, len),
                "unaligned" => (4096, len),
                "unaligned64" => (64, len),
                _ => (0, len + 4),
            };
            let ptr = unsafe { base.add(off) };
            let s = no_copy_case(&dev, &q, ptr, l);
            let ok =
                s.starts_with("buffer created") && s.contains("visible through host ptr: true");
            unsafe { std::alloc::dealloc(base, layout) };
            let verdict = match (which, ok) {
                ("aligned", true) => "ACCEPT",
                ("aligned", false) => "REJECT",
                (_, true) => "ACCEPT (outside the documented contract)",
                (_, false) => "REJECT (expected)",
            };
            result(
                &case,
                verdict,
                format!("ptr % 16384 = {}, len = {l}: {s}", ptr as usize % page),
            );
        }
        "vec" => {
            // A plain Vec from the global allocator: is a large allocation page aligned by luck?
            let len = 1 << 20;
            let mut v: Vec<u8> = vec![0u8; len];
            let ptr = v.as_mut_ptr();
            let align = ptr as usize % page;
            let s = if align == 0 {
                no_copy_case(&dev, &q, ptr, len)
            } else {
                "skipped (not 16 KB aligned)".into()
            };
            result(
                &case,
                "VALUE",
                format!("vec![0u8; 1 MiB] ptr % 16384 = {align}: {s}"),
            );
            drop(v);
        }
        _ => panic!("no-copy aligned|unaligned|unaligned64|oddlen|vec"),
    }
}

// ---------------------------------------------------------------------------------------------

fn dispatch_case(args: &[String]) {
    let a = |i: usize| args.get(i).map(String::as_str).unwrap_or("");
    match a(0) {
        "device" => probe_device(),
        "host-name" => probe_host_name(),
        "extern" => probe_extern(),
        "designated" => probe_designated(),
        "gpu-address" => probe_gpu_address(a(1) != "no-use-resource"),
        "tg-static" => probe_tg_static(a(1)),
        "tg-dynamic" => probe_tg_dynamic(a(1).parse().expect("tg-dynamic <bytes>")),
        "counters" => probe_counters(),
        "watchdog" => probe_watchdog(a(1).parse().unwrap_or(60.0)),
        "assert" => probe_assert(),
        "maxthreads" => probe_maxthreads(),
        "mul-bench" => probe_mul_bench(),
        "simd-shuffle" => probe_simd_shuffle(),
        "empty-dispatch" => probe_empty_dispatch(),
        "max-dispatches" => probe_max_dispatches(a(1).parse().expect("max-dispatches <n>")),
        "no-copy" => probe_no_copy(a(1)),
        other => {
            eprintln!("unknown probe `{other}`");
            std::process::exit(2);
        }
    }
}

const ALL: &[&[&str]] = &[
    &["device"],
    &["host-name"],
    &["extern"],
    &["designated"],
    &["gpu-address", "use-resource"],
    &["gpu-address", "no-use-resource"],
    &["tg-static", "32k"],
    &["tg-static", "36k"],
    &["tg-dynamic", "16384"],
    &["tg-dynamic", "32768"],
    &["tg-dynamic", "32784"],
    &["tg-dynamic", "36864"],
    &["tg-dynamic", "65536"],
    &["tg-dynamic", "131072"],
    &["counters"],
    &["assert"],
    &["maxthreads"],
    &["mul-bench"],
    &["simd-shuffle"],
    &["empty-dispatch"],
    &["max-dispatches", "2048"],
    &["max-dispatches", "8192"],
    &["max-dispatches", "65536"],
    &["max-dispatches", "262144"],
    &["no-copy", "aligned"],
    &["no-copy", "unaligned"],
    &["no-copy", "unaligned64"],
    &["no-copy", "oddlen"],
    &["no-copy", "vec"],
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) != Some("all") {
        if args.is_empty() {
            eprintln!("usage: runner all [--watchdog] | <probe> [arg]");
            std::process::exit(2);
        }
        dispatch_case(&args);
        return;
    }
    let mut cases: Vec<Vec<String>> = ALL
        .iter()
        .map(|c| c.iter().map(|s| s.to_string()).collect())
        .collect();
    if args.iter().any(|a| a == "--watchdog") {
        cases.push(vec!["watchdog".into(), "60".into()]);
    }
    let exe = std::env::current_exe().unwrap();
    let mut failures = 0;
    for c in cases {
        let o = Command::new(&exe).args(&c).output().unwrap();
        let stdout = String::from_utf8_lossy(&o.stdout);
        let results: Vec<&str> = stdout.lines().filter(|l| l.starts_with("RESULT")).collect();
        if results.is_empty() || !o.status.success() {
            failures += 1;
            let err = String::from_utf8_lossy(&o.stderr);
            println!(
                "RESULT {}: CRASHED | exit={:?} stderr: {}",
                c.join("-"),
                o.status,
                err.lines().rev().take(4).collect::<Vec<_>>().join(" / ")
            );
        }
        for r in results {
            println!("{r}");
        }
    }
    if failures > 0 {
        println!("{failures} case(s) crashed");
    }
}
