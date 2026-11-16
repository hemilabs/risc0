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

use std::sync::{atomic::Ordering, Arc};

use risc0_binfmt::MemoryImage;
use risc0_circuit_rv32im_sys::{
    RawMemoryTransaction, RawPreflightCycle, METAL_KERNEL_NAMES, METAL_LIB,
};
use risc0_core::field::{baby_bear::BabyBear, Elem as _, ExtElem as _};
use risc0_zkp::{
    adapter::CircuitInfo as _,
    core::hash::sha::Sha256HashSuite,
    hal::{cpu::CpuHal, metal::MetalHalSha256, Buffer as _, CircuitHal, Hal},
    INV_RATE,
};
use test_log::test;

use super::*;
use crate::{
    execute::{testutil, DEFAULT_SEGMENT_LIMIT_PO2},
    prove::hal::cpu::CpuCircuitHal,
    zirgen::{
        circuit::{Val, REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE, REGISTER_GROUP_DATA},
        info::NUM_POLY_MIX_POWERS,
        taps::TAPSET,
        CircuitImpl,
    },
    MAX_INSN_CYCLES,
};

/// The rv32im Metal tests' shared lock (PLAN 8.7), poison-tolerant like keccak's
/// `gpu_test_lock`. Taken for their whole run by the tests that change or depend on
/// process-wide state: `pipeline_bench` (environment variables between rounds, the
/// [PIPELINE_HOST_STATS] counters), the binary-archive tests (archive files, timings), so two of
/// them never overlap under the default parallel harness, e.g. with `--include-ignored`.
pub(crate) fn metal_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Whether the test harness runs one test at a time (`--test-threads 1`, `--test-threads=1` or
/// `RUST_TEST_THREADS=1`). A test that sets environment variables needs this: every other test
/// may read them (a `getenv` racing a `setenv` is a data race in libc), and a lock taken only by
/// some tests cannot exclude the readers.
pub(crate) fn harness_is_single_threaded() -> bool {
    let args: Vec<String> = std::env::args().collect();
    for (i, a) in args.iter().enumerate() {
        if a == "--test-threads" {
            return args.get(i + 1).is_some_and(|v| v.trim() == "1");
        }
        if let Some(v) = a.strip_prefix("--test-threads=") {
            return v.trim() == "1";
        }
    }
    std::env::var("RUST_TEST_THREADS").is_ok_and(|v| v.trim() == "1")
}

/// W7-R1: the lib-suite tests whose own peak process footprint is above ~2 GiB (measured one at a
/// time, notes/W7-R1.md) take this for their whole run, so the default parallel harness never
/// runs two of them at once. Same lock as [metal_test_lock] (one lock: a test that needs both
/// takes it once). It is a non-reentrant `std::sync::Mutex`: code running under it must not take
/// [metal_test_lock] again (that would deadlock).
pub(crate) fn large_test_lock() -> std::sync::MutexGuard<'static, ()> {
    metal_test_lock()
}

/// W7-R1: `R11_NO_GPU_RUN_CHECK=1` lets the heavy ignored tests run without `bin/gpu-run` (a
/// machine with no other GPU work and no lock tooling).
pub(crate) const NO_GPU_RUN_CHECK_ENV: &str = "R11_NO_GPU_RUN_CHECK";

/// W7-R1: the `bin/gpu-run` mode a heavy `#[ignore]` test needs. `Heavy` accepts the exclusive
/// lock in either of its forms (`GPU_RUN_MODE=heavy` or `huge`); `Huge` accepts only `--huge`
/// (exclusive, with a memory admission check sized for a ~20 GB process).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuRunClass {
    Heavy,
    Huge,
}

/// W7-R1: the decision of [require_gpu_run_class], without the environment: `mode` is
/// `GPU_RUN_MODE` (None if unset). Err carries the panic message, which names the actual mode.
pub(crate) fn gpu_run_check(
    test: &str,
    need: GpuRunClass,
    mode: Option<&str>,
) -> Result<(), String> {
    let hint = match need {
        GpuRunClass::Heavy => {
            "`bin/gpu-run --tag <item> -- <test command>` (or `--huge` for the \
                               CPU-plus-Metal po2 20 tests)"
        }
        GpuRunClass::Huge => "`bin/gpu-run --huge --tag <item> -- <test command>`",
    };
    let escape = format!("or set {NO_GPU_RUN_CHECK_ENV}=1 on a machine with no lock tooling");
    match mode.map(str::trim) {
        Some("huge") => Ok(()),
        Some("heavy") if need == GpuRunClass::Heavy => Ok(()),
        Some("heavy") => Err(format!(
            "{test} in this configuration is a huge-class GPU test but runs under the plain heavy \
             lock (GPU_RUN_MODE=\"heavy\"): run it alone with {hint}, whose admission check waits \
             for enough free memory ({escape})"
        )),
        Some("light") => Err(format!(
            "{test} is a heavy GPU test but runs under `gpu-run --light` (GPU_RUN_MODE=\"light\"): \
             run it alone under the exclusive lock, {hint} ({escape})"
        )),
        Some(m) => Err(format!(
            "{test} is a heavy GPU test and GPU_RUN_MODE={m:?} is not a mode bin/gpu-run sets \
             (heavy, huge, light): run it alone under the exclusive lock, {hint} ({escape})"
        )),
        None => Err(format!(
            "{test} is a heavy GPU test (several GiB of wired GPU memory) and GPU_RUN_MODE is \
             unset: run it alone under the exclusive lock, {hint}, which also sets a per-process \
             GPU budget ({escape})"
        )),
    }
}

/// W7-R1 (LEADS2 S1): the gate every heavy `#[ignore]` test runs first (through
/// [begin_heavy_test]). It panics unless the process runs under `bin/gpu-run`'s exclusive lock
/// (`GPU_RUN_MODE=heavy` or `huge`; `GpuRunClass::Huge`, for configurations that reach ~20 GB in
/// one process such as po2 21 or the old keep-every-prover modes, accepts only `huge`), which
/// also exports a per-process GPU budget, or [NO_GPU_RUN_CHECK_ENV] is `1`. Two heavy runs at
/// once, or one beside other GPU tests, exhausted the 36 GB Mac twice (wired GPU memory: a kernel
/// watchdog panic, not a killed process).
pub(crate) fn require_gpu_run_class(test: &str, need: GpuRunClass) {
    if std::env::var(NO_GPU_RUN_CHECK_ENV).is_ok_and(|v| v.trim() == "1") {
        eprintln!("[w7-r1] {test}: {NO_GPU_RUN_CHECK_ENV}=1: not checking for bin/gpu-run");
        return;
    }
    let mode = std::env::var("GPU_RUN_MODE").ok();
    if let Err(msg) = gpu_run_check(test, need, mode.as_deref()) {
        panic!("{msg}");
    }
}

/// W7-R1 fix round 2: the start of every heavy `#[ignore]` test in this module tree: the
/// `bin/gpu-run` gate ([require_gpu_run_class]), then [large_test_lock] for the test's whole run.
/// The gate checks each test alone; the lock keeps two of them (or one and a large default-suite
/// test) from running at once in one process, e.g. under `<bin> --ignored` with the default
/// parallel harness, where two ~10 GiB po2 20 workloads would otherwise overlap under the plain
/// heavy lock. A test that reruns itself in a child ([rerun_without_malloc_large_cache]) holds
/// this in the parent across the child's run (the child is another process with its own lock, so
/// this cannot deadlock).
#[must_use = "hold the lock for the whole test"]
pub(crate) fn begin_heavy_test(
    test: &str,
    need: GpuRunClass,
) -> std::sync::MutexGuard<'static, ()> {
    require_gpu_run_class(test, need);
    let t = std::time::Instant::now();
    let guard = large_test_lock();
    eprintln!(
        "[w7-r1] {test}: holding the large-test lock (waited {} ms)",
        t.elapsed().as_millis()
    );
    guard
}

/// W7-R1 fix round 1: the review found `gpu-run --huge` (GPU_RUN_MODE=huge) rejected with a
/// message that said `--light`. The gate must accept heavy and huge for heavy tests, only huge for
/// huge-class configurations, and name the actual mode when it refuses.
#[test]
fn gpu_run_check_modes() {
    use GpuRunClass::*;
    assert!(gpu_run_check("t", Heavy, Some("heavy")).is_ok());
    assert!(gpu_run_check("t", Heavy, Some("huge")).is_ok());
    assert!(gpu_run_check("t", Huge, Some("huge")).is_ok());
    let e = gpu_run_check("t", Huge, Some("heavy")).unwrap_err();
    assert!(
        e.contains("\"heavy\"") && e.contains("--huge") && !e.contains("--light"),
        "{e}"
    );
    for need in [Heavy, Huge] {
        let e = gpu_run_check("t", need, Some("light")).unwrap_err();
        assert!(e.contains("--light") && e.contains("\"light\""), "{e}");
        let e = gpu_run_check("t", need, None).unwrap_err();
        assert!(e.contains("unset") && !e.contains("--light"), "{e}");
        let e = gpu_run_check("t", need, Some("bogus")).unwrap_err();
        assert!(e.contains("\"bogus\"") && !e.contains("--light"), "{e}");
    }
}

/// W7-R1 fix round 2 (review: `<bin> --ignored` ran prove_po2_sweep beside
/// seal_matches_cpu_large's child under the plain heavy lock): every `#[ignore]` test in the
/// files this item owns must start with [begin_heavy_test] (gate, then the large-test lock), and
/// a test that reruns itself in a child must take the lock before it spawns the child.
#[test]
fn ignored_heavy_tests_take_the_large_test_lock() {
    let files = [
        ("integration_tests.rs", include_str!("integration_tests.rs")),
        ("witgen_tests.rs", include_str!("witgen_tests.rs")),
        ("pipeline_bench.rs", include_str!("pipeline_bench.rs")),
        ("tests.rs", include_str!("tests.rs")),
    ];
    let mut found = Vec::new();
    for (file, src) in files {
        let lines: Vec<&str> = src.lines().collect();
        for (i, l) in lines.iter().enumerate() {
            if !l.starts_with("#[ignore") {
                continue;
            }
            let f = (i + 1..lines.len())
                .find(|&j| lines[j].starts_with("fn "))
                .unwrap_or_else(|| panic!("{file}:{}: #[ignore] without a fn", i + 1));
            let name = lines[f][3..].split('(').next().unwrap().to_string();
            let end = (f..lines.len())
                .find(|&j| lines[j] == "}")
                .unwrap_or_else(|| panic!("{file}: {name}: no closing brace"));
            let body = lines[f..end].join("\n");
            let lock = body.find("begin_heavy_test(").unwrap_or_else(|| {
                panic!("{file}: {name} is #[ignore] but does not call begin_heavy_test")
            });
            if let Some(child) = body.find("rerun_without_malloc_large_cache(") {
                assert!(
                    lock < child,
                    "{file}: {name} spawns its child before it takes the large-test lock"
                );
            }
            found.push(name);
        }
    }
    for t in [
        "prove_po2_sweep",
        "seal_matches_cpu_large",
        "r15w_witgen_ab",
        "pipeline_bench",
        "archive_cold_start",
    ] {
        assert!(found.iter().any(|f| f == t), "{t} not found: {found:?}");
    }
}

/// W7-R1: macOS malloc keeps freed large blocks in its "large cache" (`vmmap`: `Malloc Large
/// (empty)`, dirty, counted in the footprint): a dropped CPU prover's po2 18 / 19 buffers stay
/// as +2.2 / +5.5 GiB of footprint under the next Metal proof, and `malloc_zone_pressure_relief`
/// does not release them. `MallocLargeCache=0` (read by libmalloc at process start) turns the
/// cache off. A heavy test that runs the CPU prover beside Metal proofs calls this first: unless
/// that is already set (or `R11_KEEP_MALLOC_CACHE=1`), it runs `test` (its full path in the test
/// binary) again in a child process with `MallocLargeCache=0`, inheriting the gpu-run lock, and
/// returns true; the caller then returns (the child did the work and its exit status is checked).
pub(crate) fn rerun_without_malloc_large_cache(test: &str) -> bool {
    if std::env::var("MallocLargeCache").is_ok_and(|v| v.trim() == "0")
        || std::env::var("R11_KEEP_MALLOC_CACHE").is_ok_and(|v| v.trim() == "1")
    {
        return false;
    }
    let name = test.split_once("::").map_or(test, |(_, rest)| rest);
    eprintln!(
        "[w7-r1] {name}: running again with MallocLargeCache=0 (the CPU prover's freed buffers \
         otherwise stay in the footprint; R11_KEEP_MALLOC_CACHE=1 skips this)"
    );
    let exe = std::env::current_exe().expect("test binary path");
    let mut child = std::process::Command::new(exe)
        .args([
            name,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("MallocLargeCache", "0")
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("run the test binary");
    // Echo the child's stdout (the harness summary) and keep it to check that the test ran.
    let mut out = String::new();
    {
        use std::io::{BufRead as _, Write as _};
        let reader = std::io::BufReader::new(child.stdout.take().unwrap());
        for line in reader.lines() {
            let line = line.expect("child stdout");
            println!("{line}");
            let _ = std::io::stdout().flush();
            out.push_str(&line);
            out.push('\n');
        }
    }
    // `/usr/bin/time -l` around this process sees only the parent (~2.3 MB): read the child's
    // lifetime peak footprint while it is a zombie (waitid WNOWAIT), then reap it.
    let child_peak = exited_child_peak_footprint(child.id());
    let status = child.wait().expect("wait for the test binary");
    match child_peak {
        Some(p) => eprintln!(
            "[w7-r1] {name}: child (MallocLargeCache=0) peak footprint {p} B ({:.2} GiB); \
             `time -l` on this process reports only the parent",
            p as f64 / (1u64 << 30) as f64
        ),
        None => eprintln!("[w7-r1] {name}: child peak footprint unavailable"),
    }
    assert!(status.success(), "{name} with MallocLargeCache=0: {status}");
    assert!(
        child_ran_one_test(&out),
        "{name} with MallocLargeCache=0: the child did not run exactly one test (the filter \
         matched nothing, or the caller is not #[ignore]); its stdout:\n{out}"
    );
    true
}

/// W7-R1: whether a test binary's stdout shows exactly one test run and passed (not a vacuous
/// pass of a filter that matched nothing).
pub(crate) fn child_ran_one_test(stdout: &str) -> bool {
    stdout
        .lines()
        .any(|l| l.starts_with("test result: ok. 1 passed;"))
}

#[test]
fn child_ran_one_test_rejects_empty_filter() {
    assert!(child_ran_one_test(
        "running 1 test\ntest a::b ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 \
         measured; 140 filtered out; finished in 1.00s\n"
    ));
    assert!(!child_ran_one_test(
        "running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 141 \
         filtered out; finished in 0.00s\n"
    ));
}

/// W7-R1: wait (without reaping) for the child `pid` to exit and return its lifetime peak
/// physical footprint (`proc_pid_rusage` on the zombie); None if either call fails.
fn exited_child_peak_footprint(pid: u32) -> Option<u64> {
    extern "C" {
        fn waitid(idtype: i32, id: u32, infop: *mut std::ffi::c_void, options: i32) -> i32;
    }
    const P_PID: i32 = 1;
    const WEXITED: i32 = 0x04;
    const WNOWAIT: i32 = 0x20;
    // siginfo_t is 104 bytes on Darwin.
    let mut info = [0u64; 16];
    loop {
        // SAFETY: infop points to 128 writable bytes, more than a siginfo_t.
        let rc = unsafe { waitid(P_PID, pid, info.as_mut_ptr().cast(), WEXITED | WNOWAIT) };
        if rc == 0 {
            break;
        }
        if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
            return None;
        }
    }
    pid_footprint(pid as i32).map(|(_, peak)| peak)
}

/// W7-R1: this process' physical footprint and its lifetime peak (`proc_pid_rusage`,
/// `RUSAGE_INFO_V4`), bytes; zeros if the call fails. Unlike RSS the footprint counts the GPU
/// memory charged to the process: MTLBuffers, and the memory the driver adds when a pipeline with a
/// deep native stack is first dispatched, which `currentAllocatedSize` does not count.
pub(crate) fn process_footprint() -> (u64, u64) {
    pid_footprint(std::process::id() as i32).unwrap_or((0, 0))
}

/// `(ri_phys_footprint, ri_lifetime_max_phys_footprint)` of `pid`, bytes.
fn pid_footprint(pid: i32) -> Option<(u64, u64)> {
    extern "C" {
        fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut std::ffi::c_void) -> i32;
    }
    // rusage_info_v4: a 16-byte uuid, then u64 counters; ri_phys_footprint is counter 7 (u64
    // index 9) and ri_lifetime_max_phys_footprint counter 28 (u64 index 30); 37 u64s in all.
    let mut info = [0u64; 40];
    // SAFETY: RUSAGE_INFO_V4 (4) writes a rusage_info_v4 (296 bytes) into the 320-byte buffer.
    let rc = unsafe { proc_pid_rusage(pid, 4, info.as_mut_ptr().cast()) };
    if rc != 0 {
        return None;
    }
    Some((info[9], info[30]))
}

#[test]
fn missing_kernels_error_only_when_phase_required() {
    let hal = MetalHalSha256::new();
    let expected: &[(Phase, &[&str])] = &[
        (Phase::Witgen, &["injectCycleCols"]),
        (Phase::Accum, &["r7_not_a_kernel", "finalizeZeroizeAccum"]),
        (Phase::EvalCheck, &[]),
    ];
    let k = Rv32imKernels::load_from(&hal, METAL_LIB, METAL_KERNEL_NAMES, expected).unwrap();
    k.require(Phase::Witgen).unwrap();
    k.require(Phase::EvalCheck).unwrap();
    let err = format!("{:#}", k.require(Phase::Accum).unwrap_err());
    assert!(
        err.contains("Accum") && err.contains("r7_not_a_kernel"),
        "{err}"
    );
    assert!(!err.contains("finalizeZeroizeAccum"), "{err}");
    // Every present listed/expected kernel has a PSO.
    assert!(k.kernels.get("injectCycleCols").is_some());
    assert!(k.kernels.get("finalizeZeroizeAccum").is_some());
}

#[test]
fn listed_kernel_missing_fails_load() {
    let hal = MetalHalSha256::new();
    let err = Rv32imKernels::load_from(&hal, METAL_LIB, &["r7_not_a_kernel"], &[])
        .err()
        .expect("a name listed by rv32im-sys must exist");
    assert!(format!("{err:#}").contains("r7_not_a_kernel"));
}

#[test]
fn real_metallib_loads() {
    let hal = Arc::new(MetalHalPoseidon2::new());
    let circuit_hal = MetalCircuitHalPoseidon2::new(hal).unwrap();
    for name in METAL_KERNEL_NAMES {
        let _ = circuit_hal.ctx().pipeline(name);
    }
    // Each phase either has all its kernels or reports which are missing.
    for &(phase, names) in EXPECTED_KERNELS {
        if let Err(e) = circuit_hal.ctx().require(phase) {
            let e = format!("{e:#}");
            assert!(names.iter().any(|n| e.contains(n)), "{e}");
        }
    }
    warmup().unwrap();
}

#[test]
fn scratch_tables_and_preflight_upload() {
    let hal = Arc::new(MetalHalSha256::new());
    let ctx = MetalRv32imCtx::new(hal.clone()).unwrap();
    let (u8t, u16t) = ctx.tables();
    assert!(u8t.size() >= TABLE_U8_CELLS && u16t.size() >= TABLE_U16_CELLS);
    let (u8b, _) = ctx.tables();
    assert_eq!(
        u8b.as_device_ptr(),
        u8t.as_device_ptr(),
        "tables are reused"
    );

    let mut trace = PreflightTrace::default();
    for i in 0..37u32 {
        trace.cycles.push(RawPreflightCycle {
            state: i,
            pc: 4 * i,
            major: (i % 7) as u8,
            minor: (i % 5) as u8,
            machine_mode: 1,
            padding: 0,
            user_cycle: i + 1,
            txn_idx: i + 2,
            paging_idx: i + 3,
            bigint_idx: i + 4,
            diff_count: [i + 5, i + 6],
        });
    }
    for i in 0..11u32 {
        trace.txns.push(RawMemoryTransaction {
            addr: i,
            cycle: i * 2,
            word: i * 3,
            prev_cycle: i * 4,
            prev_word: i * 5,
        });
    }
    trace.bigint_bytes = (0..13u8).collect();
    let (buf, layout) = ctx.upload_preflight(&trace).unwrap();
    assert_eq!(layout, PreflightLayout::new(37, 11, 13));
    assert_eq!(layout.txns_offset % PF_ALIGN, 0);
    assert_eq!(layout.bigint_offset % PF_ALIGN, 0);
    assert!(layout.txns_offset >= 37 * PREFLIGHT_CYCLE_BYTES);
    assert!(buf.size() >= layout.bytes);
    let bytes = buf.to_vec();
    let cyc = &bytes[layout.cycles_offset..][..37 * PREFLIGHT_CYCLE_BYTES];
    // cycle 5: state=5, pc=20, major=5, minor=0, mm=1, pad, user_cycle=6, ... diff_count[1]=11
    let c5 = &cyc[5 * PREFLIGHT_CYCLE_BYTES..][..PREFLIGHT_CYCLE_BYTES];
    assert_eq!(&c5[0..4], &5u32.to_le_bytes());
    assert_eq!(&c5[4..8], &20u32.to_le_bytes());
    assert_eq!(&c5[8..12], &[5, 0, 1, 0]);
    assert_eq!(&c5[32..36], &11u32.to_le_bytes());
    let t10 = &bytes[layout.txns_offset + 10 * MEMORY_TXN_BYTES..][..MEMORY_TXN_BYTES];
    assert_eq!(&t10[16..20], &50u32.to_le_bytes());
    assert_eq!(
        &bytes[layout.bigint_offset..][..13],
        trace.bigint_bytes.as_slice()
    );
    // A smaller trace reuses the buffer.
    trace.cycles.truncate(3);
    let (buf2, _) = ctx.upload_preflight(&trace).unwrap();
    assert_eq!(buf2.as_device_ptr(), buf.as_device_ptr());
}

fn random_vals(n: usize) -> Vec<Val> {
    let mut rng = rand::rng();
    (0..n).map(|_| Val::random(&mut rng)).collect()
}

/// Runs `CircuitHal::eval_check` of `c` on `hal` over copies of `bufs` and returns `check`.
fn run_eval_check<H, C>(
    hal: &H,
    c: &C,
    bufs: [&[Val]; 5],
    poly_mix: ExtVal,
    po2: usize,
    steps: usize,
) -> Vec<Val>
where
    H: Hal<Elem = Val, ExtElem = ExtVal>,
    C: CircuitHal<H>,
{
    let check = hal.alloc_elem("check", ExtVal::EXT_SIZE * steps * INV_RATE);
    let [code, data, accum, mix, out] = bufs.map(|b| hal.copy_from_elem("in", b));
    c.eval_check(
        &check,
        &[&accum, &code, &data],
        &[&mix, &out],
        poly_mix,
        po2,
        steps,
    );
    c.eval_check_dep();
    check.to_vec()
}

/// R10: the Metal eval_check (baseline `eval_check` kernel) is bit-exact with `CpuCircuitHal`'s on
/// random canonical inputs (`Val::random` is uniform in [0, P)), as hal/cuda.rs tests CUDA. Also
/// checks the CPU fallback (`RISC0_METAL_RV32IM_CPU_PHASES=eval`) and the `CircuitHal` entry
/// point, and reports the GPU memory of the phase (PLAN 9.1).
fn eval_check_matches_cpu(po2: usize) {
    let steps = 1 << po2;
    let domain = steps * INV_RATE;
    let code = random_vals(TAPSET.group_size(REGISTER_GROUP_CODE) * domain);
    let data = random_vals(TAPSET.group_size(REGISTER_GROUP_DATA) * domain);
    let accum = random_vals(TAPSET.group_size(REGISTER_GROUP_ACCUM) * domain);
    let mix = random_vals(CircuitImpl::MIX_SIZE);
    let out = random_vals(CircuitImpl::OUTPUT_SIZE);
    let poly_mix = ExtVal::random(&mut rand::rng());
    let bufs = [&code[..], &data, &accum, &mix, &out];

    let t = std::time::Instant::now();
    let cpu_hal: CpuHal<BabyBear> = CpuHal::new(Sha256HashSuite::new_suite());
    let cpu = run_eval_check(&cpu_hal, &CpuCircuitHal, bufs, poly_mix, po2, steps);
    let cpu_time = t.elapsed();

    let hal = Arc::new(MetalHalSha256::new());
    let metal_c = MetalCircuitHal::new(hal.clone()).unwrap();
    let check = hal.alloc_elem("check", ExtVal::EXT_SIZE * domain);
    let [code_b, data_b, accum_b, mix_b, out_b] = bufs.map(|b| hal.copy_from_elem("in", b));
    let groups = [&accum_b, &code_b, &data_b];
    let globals = [&mix_b, &out_b];
    hal.sync();
    hal.reset_peak_allocated_size();
    let before = hal.allocated_size();
    let t = std::time::Instant::now();
    metal_c
        .eval_check_metal(&check, &groups, &globals, poly_mix, po2, steps)
        .unwrap();
    let gpu_time = t.elapsed();
    let peak = hal.peak_allocated_size().max(hal.allocated_size());
    // W7-R1: the footprint also holds what the driver adds for a deep-stack pipeline, which
    // currentAllocatedSize does not count (the baseline kernel: ~8 GB at any po2; notes/W7-R1.md).
    let (footprint, _) = process_footprint();
    let gpu = check.to_vec();
    let mismatches = cpu.iter().zip(&gpu).filter(|(a, b)| a != b).count();
    assert_eq!(
        mismatches,
        0,
        "po2 {po2}: {mismatches} of {} differ",
        cpu.len()
    );
    assert_eq!(cpu, gpu);
    let inputs: usize = [&code[..], &data, &accum, &mix, &out]
        .iter()
        .map(|b| b.len() * 4)
        .sum();
    println!(
        "R10 eval_check po2 {po2}: domain {domain}, bit-exact {} elements; GPU {gpu_time:?}, \
         CPU HAL {cpu_time:?}; currentAllocatedSize before phase {before} B, peak during phase \
         {peak} B (phase buffers: inputs {inputs} B + check {} B + poly_mix {} B); process \
         footprint after the phase {footprint} B",
        gpu.len(),
        gpu.len() * 4,
        NUM_POLY_MIX_POWERS * 16,
    );

    // The CPU fallback over the same Metal buffers.
    let check2 = hal.alloc_elem("check", ExtVal::EXT_SIZE * domain);
    metal_c
        .eval_check_cpu(&check2, &groups, &globals, poly_mix, po2, steps)
        .unwrap();
    assert_eq!(cpu, check2.to_vec(), "CPU fallback, po2 {po2}");

    // The CircuitHal entry point (Metal unless RISC0_METAL_RV32IM_CPU_PHASES selects eval).
    if po2 <= 10 {
        let calls = PHASE_CALLS[Phase::EvalCheck as usize].load(Ordering::Relaxed);
        let via_trait = run_eval_check(hal.as_ref(), &metal_c, bufs, poly_mix, po2, steps);
        assert_eq!(cpu, via_trait, "CircuitHal::eval_check, po2 {po2}");
        assert!(PHASE_CALLS[Phase::EvalCheck as usize].load(Ordering::Relaxed) > calls);
    }
}

#[test]
fn eval_check_matches_cpu_po2_4() {
    eval_check_matches_cpu(4);
}

#[test]
fn eval_check_matches_cpu_po2_10() {
    eval_check_matches_cpu(10);
}

#[test]
fn eval_check_matches_cpu_po2_16() {
    eval_check_matches_cpu(16);
}

/// The same poly_mix buffer is reused across calls with different poly_mix values and po2s.
#[test]
fn eval_check_reuses_poly_mix_buffer() {
    let hal = Arc::new(MetalHalSha256::new());
    let metal_c = MetalCircuitHal::new(hal.clone()).unwrap();
    let cpu_hal: CpuHal<BabyBear> = CpuHal::new(Sha256HashSuite::new_suite());
    let mut pm_ptr = None;
    for po2 in [5, 4, 6] {
        let steps = 1 << po2;
        let domain = steps * INV_RATE;
        let code = random_vals(TAPSET.group_size(REGISTER_GROUP_CODE) * domain);
        let data = random_vals(TAPSET.group_size(REGISTER_GROUP_DATA) * domain);
        let accum = random_vals(TAPSET.group_size(REGISTER_GROUP_ACCUM) * domain);
        let mix = random_vals(CircuitImpl::MIX_SIZE);
        let out = random_vals(CircuitImpl::OUTPUT_SIZE);
        let poly_mix = ExtVal::random(&mut rand::rng());
        let bufs = [&code[..], &data, &accum, &mix, &out];
        let cpu = run_eval_check(&cpu_hal, &CpuCircuitHal, bufs, poly_mix, po2, steps);
        // eval_check_metal directly, so the test holds under RISC0_METAL_RV32IM_CPU_PHASES=eval.
        let check = hal.alloc_elem("check", ExtVal::EXT_SIZE * domain);
        let [code_b, data_b, accum_b, mix_b, out_b] = bufs.map(|b| hal.copy_from_elem("in", b));
        metal_c
            .eval_check_metal(
                &check,
                &[&accum_b, &code_b, &data_b],
                &[&mix_b, &out_b],
                poly_mix,
                po2,
                steps,
            )
            .unwrap();
        assert_eq!(cpu, check.to_vec(), "po2 {po2}");
        let ptr = metal_c
            .ctx()
            .scratch()
            .poly_mix
            .as_ref()
            .unwrap()
            .as_device_ptr();
        assert_eq!(*pm_ptr.get_or_insert(ptr), ptr, "poly_mix buffer reused");
    }
}

#[test]
fn cpu_phases_list_parsing() {
    use super::phase_listed;
    assert!(phase_listed("eval", "eval"));
    assert!(phase_listed("witgen, Eval ,accum", "eval"));
    assert!(!phase_listed("witgen,accum", "eval"));
    assert!(!phase_listed("", "eval"));
    assert!(!phase_listed("evaluate", "eval"));
}

/// The default rv32im segment prover of a `metal` build is the Metal one: proving a segment
/// goes through every phase of [MetalCircuitHal], and the seal verifies.
#[test]
fn segment_prover_uses_metal_hal() {
    assert!(
        risc0_zkp::hal::metal::enabled("rv32im"),
        "RISC0_METAL / RISC0_METAL_SKIP disable rv32im"
    );
    let before: Vec<usize> = PHASE_CALLS
        .iter()
        .map(|c| c.load(Ordering::Relaxed))
        .collect();
    let image = MemoryImage::new_kernel(testutil::kernel::basic());
    let result = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        MAX_INSN_CYCLES,
        testutil::DEFAULT_SESSION_LIMIT,
        &testutil::NullSyscall,
        None,
    )
    .unwrap();
    let segment = result.segments.first().unwrap();
    let prover = crate::prove::segment_prover().unwrap();
    let seal = prover.prove(segment).unwrap();
    crate::verify(&seal).unwrap();
    for (phase, (c, b)) in [Phase::Witgen, Phase::Accum, Phase::EvalCheck]
        .iter()
        .zip(PHASE_CALLS.iter().zip(before))
    {
        assert!(
            c.load(Ordering::Relaxed) > b,
            "{phase:?} did not run on the Metal circuit HAL"
        );
    }
}

/// A fresh directory for one test's binary archive (removed at the start).
fn archive_temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("risc0-r14-archive-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// R14: the rv32im PSOs through the zkp binary archive. A first load into an empty directory
/// misses every kernel and writes the archive; a second load hits every kernel and writes
/// nothing; both have a PSO for every requested kernel, with the same thread limits. Small,
/// quick kernels only (the full set is `archive_cold_start`). The loader is chosen by parameter,
/// never through [ARCHIVE_ENV]; `RISC0_METAL_ARCHIVE=0` set for the whole process (all units)
/// leaves no archive, and the test then only checks the load.
fn archive_roundtrip(hal: &MetalHalSha256, dir: &std::path::Path, abs_dir: &std::path::Path) {
    let names = ["injectCycleCols", "finalizeZeroizeAccum", "ef_combine"];
    let expected: &[(Phase, &[&str])] = &[];
    let load = || {
        Rv32imKernels::load_from_in(
            hal,
            METAL_LIB,
            &names,
            expected,
            true,
            Some(dir.to_path_buf()),
            "rv32im",
        )
        .unwrap()
    };
    let first = load();
    let Some(r) = first.kernels.archive_report() else {
        panic!("the archived loader must report");
    };
    if r.path.is_none() && r.errors.is_empty() {
        return; // archive disabled for the process (or the OS build is unknown)
    }
    assert!(r.errors.is_empty(), "{r:?}");
    assert_eq!(
        (r.hits, r.misses, r.loaded, r.saved),
        (0, names.len(), false, true),
        "{r:?}"
    );
    let path = r.path.clone().unwrap();
    assert!(path.starts_with(abs_dir), "{path:?} not in {abs_dir:?}");
    let saved: Vec<_> = std::fs::read_dir(abs_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".binarchive"))
        .collect();
    assert_eq!(saved.len(), 1, "one archive file in {abs_dir:?}");
    let second = load();
    let r = second.kernels.archive_report().unwrap();
    assert!(r.errors.is_empty(), "{r:?}");
    assert_eq!(
        (r.hits, r.misses, r.loaded, r.saved),
        (names.len(), 0, true, false),
        "{r:?}"
    );
    for name in names {
        let (a, b) = (first.kernels.pipeline(name), second.kernels.pipeline(name));
        assert_eq!(
            a.max_total_threads_per_threadgroup(),
            b.max_total_threads_per_threadgroup(),
            "{name}"
        );
        assert_eq!(
            a.thread_execution_width(),
            b.thread_execution_width(),
            "{name}"
        );
    }
    // The unloaded kernels are still reachable through the library (lazy PSOs).
    assert!(second
        .kernels
        .library()
        .function_names()
        .iter()
        .any(|f| f == "eval_check"));
}

#[test]
fn archive_roundtrip_small_set() {
    let _lock = metal_test_lock();
    let hal = MetalHalSha256::new();
    let dir = archive_temp_dir("small");
    archive_roundtrip(&hal, &dir, &dir);
    let _ = std::fs::remove_dir_all(&dir);
}

/// R14 fix 1: a relative archive directory (`RISC0_METAL_ARCHIVE_DIR=rel`) is resolved against
/// the current directory; before, the archive's `file://rel/..` URL could not be opened and
/// nothing was ever saved. The relative path is built from the current directory to a temp
/// directory, so the test changes no process-wide state (no `chdir`).
#[test]
fn archive_roundtrip_relative_dir() {
    let _lock = metal_test_lock();
    let hal = MetalHalSha256::new();
    let abs = archive_temp_dir("relative");
    let cwd = std::env::current_dir().unwrap();
    let mut rel = std::path::PathBuf::new();
    for _ in cwd.components().skip(1) {
        rel.push("..");
    }
    rel.push(abs.strip_prefix("/").unwrap());
    assert!(rel.is_relative());
    archive_roundtrip(&hal, &rel, &std::path::absolute(&rel).unwrap());
    let _ = std::fs::remove_dir_all(&abs);
}

/// R14 measurement: cold and warm PSO creation of the full rv32im set ([METAL_KERNEL_NAMES] and
/// [EXPECTED_KERNELS]): the plain loader, then the archive into an empty directory (misses), then
/// the archive again (hits). Run with `MTL_SHADER_CACHE_SIZE=0` (OS shader cache off) to see
/// cold times: `<bin> archive_cold_start --ignored --nocapture --test-threads 1`. The loader is
/// chosen by parameter (no environment change); the test lock keeps the other archive tests and
/// the pipeline bench out of the timings.
#[test]
#[ignore = "measurement: cold PSO builds take minutes; run alone"]
fn archive_cold_start() {
    let _lock = begin_heavy_test("archive_cold_start", GpuRunClass::Heavy);
    let hal = MetalHalSha256::new();
    let dir = archive_temp_dir("cold");
    let order = std::env::var("R14_ARCHIVE_ORDER").unwrap_or_else(|_| "plain,miss,hit".into());
    for step in order.split(',') {
        let t = std::time::Instant::now();
        let plain = step == "plain";
        let k = Rv32imKernels::load_from_in(
            &hal,
            METAL_LIB,
            METAL_KERNEL_NAMES,
            EXPECTED_KERNELS,
            !plain,
            (!plain).then(|| dir.clone()),
            "rv32im",
        )
        .unwrap();
        eprintln!(
            "[r14] archive {step}: {:.2} s, report {:?}",
            t.elapsed().as_secs_f64(),
            k.kernels.archive_report()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// R14 (R8 open issue): a panic while the scratch lock is held (e.g. a working-set budget panic
/// in `grow`) does not make every later phase of the circuit HAL panic on a poisoned Mutex.
#[test]
fn scratch_survives_a_poisoned_lock() {
    let hal = Arc::new(MetalHalSha256::new());
    let ctx = MetalRv32imCtx::new(hal).unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = ctx.scratch();
        panic!("injected panic while holding the scratch lock");
    }));
    assert!(r.is_err());
    let (u8t, u16t) = ctx.tables();
    assert!(u8t.size() >= TABLE_U8_CELLS && u16t.size() >= TABLE_U16_CELLS);
    let _ = ctx.upload_preflight(&PreflightTrace::default()).unwrap();
}
