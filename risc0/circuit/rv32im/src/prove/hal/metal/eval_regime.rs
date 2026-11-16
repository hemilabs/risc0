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

//! R22: the eval_check compiler-regime guard (docs/metal-port/notes/R22.md).
//!
//! R18 / R21: the time of the nested-chain evalfast split kernel follows the native per-thread
//! stack the Apple GPU back end gives it, in two clusters with nothing between them: ~18-21 KB
//! (the fast regime: `ef_eval_split_nest`, `ef_eval_split_fc_nest`, the defaults before R22) and
//! 32-40 KB (~2.4x slower, e.g. `ef_eval_split_r15e`). R22 measured that the stack size is a
//! symptom, not the cause (padding the stack by 16 KB costs < 1 %); the flat chain (R22's
//! default) has a ~4 KB stack and is ~7 % faster. The guard's limit separates both fast forms
//! from the slow cluster. eval_check is ~60 % of a segment, so an OS or toolchain update that
//! moves the kernel into the slow cluster would cost ~+45 % STARK time without any error.
//!
//! The binary archives the process already writes (zkp `archive`) hold the native code the OS
//! driver compiled, and its metadata has that stack size (the same FlatBuffers field
//! `tools/metal_audit/agx_size.py` reads offline). This module reads it ([native_info]) for:
//! - the plain `ef_eval_split`, once per process on a background thread at library load
//!   ([start]): a loud warning if it is in the slow regime, and with [GuardMode::Fallback] (the
//!   default) background builds of the fallback kernels ([fallback_kernels]: R22's unhoisted
//!   flat chain, then the nested chain, then R15e's form; W7-R1) through archives of their own;
//!   the first in the fast regime replaces the plain PSO ([fallback_split]);
//! - each domain-specialized `ef_eval_split_fc` PSO when it is built (eval_check.rs
//!   `build_fc`): a slow one is not used while the plain PSO is not slow ([fc_acceptable]).
//!
//! Nothing here changes a result: every candidate kernel is bit-exact with the CPU (tests). The
//! metadata layout is empirical; anything not understood is "unknown" and changes nothing.

use std::{path::Path, sync::OnceLock};

/// `RISC0_METAL_RV32IM_EVAL_GUARD`: `1` (default; check, warn and fall back), `warn` (check and
/// warn only) or `0` (off: no check, no fallback). `force` (testing only) runs the fallback
/// path as if the plain kernel were slow and uses the fallback whatever its regime. Read once per
/// process.
pub(crate) const EVAL_GUARD_ENV: &str = "RISC0_METAL_RV32IM_EVAL_GUARD";
/// `RISC0_METAL_RV32IM_EVAL_STACK_MAX`: the largest native per-thread stack (bytes) of the fast
/// regime (default [STACK_MAX]). Read once per process.
pub(crate) const EVAL_STACK_MAX_ENV: &str = "RISC0_METAL_RV32IM_EVAL_STACK_MAX";
/// Between the clusters R21 measured (fast <= 20.8 KB, slow >= 32 KB) on the M5 Max.
pub(crate) const STACK_MAX: u32 = 24576;

/// The modes of [EVAL_GUARD_ENV].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GuardMode {
    Off,
    Warn,
    Fallback,
    /// Testing: exercise the fallback path (always use the fallback PSO).
    Force,
}

impl GuardMode {
    pub(crate) fn parse(v: &str) -> anyhow::Result<Self> {
        match v.trim().to_ascii_lowercase().as_str() {
            "" | "1" | "on" | "fallback" => Ok(Self::Fallback),
            "warn" => Ok(Self::Warn),
            "0" | "off" => Ok(Self::Off),
            "force" => Ok(Self::Force),
            _ => anyhow::bail!("{EVAL_GUARD_ENV}={v:?}: expected `1`, `warn`, `0` or `force`"),
        }
    }
}

/// The guard's settings for this process (an invalid value is logged and the default used: the
/// guard never fails a proof).
pub(crate) fn settings() -> (GuardMode, u32) {
    static S: OnceLock<(GuardMode, u32)> = OnceLock::new();
    *S.get_or_init(|| {
        let mode = std::env::var(EVAL_GUARD_ENV)
            .ok()
            .map_or(Ok(GuardMode::Fallback), |v| GuardMode::parse(&v))
            .unwrap_or_else(|e| {
                tracing::warn!("{e:#}; using the default");
                GuardMode::Fallback
            });
        let max = match std::env::var(EVAL_STACK_MAX_ENV) {
            Ok(v) => v.trim().parse().unwrap_or_else(|_| {
                tracing::warn!("{EVAL_STACK_MAX_ENV}={v:?} is not a byte count; using {STACK_MAX}");
                STACK_MAX
            }),
            Err(_) => STACK_MAX,
        };
        (mode, max)
    })
}

/// Native code size and per-thread stack of one pipeline in a binary archive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NativeInfo {
    pub text: usize,
    pub stack: u32,
}

/// What the guard knows about a kernel's compile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Regime {
    Fast(NativeInfo),
    Slow(NativeInfo),
    Unknown,
}

impl Regime {
    pub(crate) fn of(info: Option<NativeInfo>, max: u32) -> Self {
        match info {
            Some(i) if i.stack <= max => Self::Fast(i),
            Some(i) => Self::Slow(i),
            None => Self::Unknown,
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Fast(i) => format!(
                "native stack {} B, {} B of code: fast regime",
                i.stack, i.text
            ),
            Self::Slow(i) => format!(
                "native stack {} B, {} B of code: SLOW regime",
                i.stack, i.text
            ),
            Self::Unknown => "unknown (no archive, or its layout is not understood)".into(),
        }
    }
}

// ---- Binary archive parsing (mirrors tools/metal_audit/agx_size.py; empirical layout) ----

const FAT_MAGICS: [u32; 2] = [0xCAFE_BABE, 0xCBFE_BABE];
const CPU_APPLEGPU: u32 = 0x0100_0013;
const MH_MAGIC_64: u32 = 0xFEED_FACF;
const LC_SEGMENT_64: u32 = 0x19;
/// The native metadata FlatBuffers: root -> field 0 (table) -> field 14 = per-thread stack bytes
/// including the deepest call chain (R21's calibration: a private `uint[N]` reports 4N + 16).
const META_STACK_FIELD: usize = 14;

fn u32_le(d: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        d.get(o..o.checked_add(4)?)?.try_into().ok()?,
    ))
}
fn u32_be(d: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_be_bytes(
        d.get(o..o.checked_add(4)?)?.try_into().ok()?,
    ))
}
fn u16_le(d: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        d.get(o..o.checked_add(2)?)?.try_into().ok()?,
    ))
}
fn u64_le(d: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        d.get(o..o.checked_add(8)?)?.try_into().ok()?,
    ))
}

/// One section of a 64-bit Mach-O image: (segment, section, file offset, size).
struct Section<'a> {
    seg: &'a [u8],
    sect: &'a [u8],
    data: &'a [u8],
}

fn cstr(b: &[u8]) -> &[u8] {
    b.split(|&c| c == 0).next().unwrap_or(b)
}

fn sections(img: &[u8]) -> Option<Vec<Section<'_>>> {
    if u32_le(img, 0)? != MH_MAGIC_64 {
        return None;
    }
    let ncmds = u32_le(img, 16)?;
    let mut out = Vec::new();
    let mut off = 32usize;
    for _ in 0..ncmds {
        let cmd = u32_le(img, off)?;
        let size = u32_le(img, off + 4)? as usize;
        if size < 8 {
            return None;
        }
        if cmd == LC_SEGMENT_64 {
            let nsects = u32_le(img, off + 64)?;
            let mut s = off + 72;
            for _ in 0..nsects {
                let sect = cstr(img.get(s..s + 16)?);
                let seg = cstr(img.get(s + 16..s + 32)?);
                let len = u64_le(img, s + 40)? as usize;
                let foff = u32_le(img, s + 48)? as usize;
                let data = img.get(foff..foff.checked_add(len)?)?;
                out.push(Section { seg, sect, data });
                s += 80;
            }
        }
        off = off.checked_add(size)?;
    }
    Some(out)
}

/// The slices of a fat image (binary archives use the magic 0xCBFEBABE); a thin image is one
/// slice of unknown CPU type.
fn slices(blob: &[u8]) -> Vec<(Option<u32>, &[u8])> {
    let Some(magic) = u32_be(blob, 0) else {
        return vec![];
    };
    if !FAT_MAGICS.contains(&magic) {
        return vec![(None, blob)];
    }
    let n = u32_be(blob, 4).unwrap_or(0) as usize;
    (0..n.min(64))
        .filter_map(|i| {
            let e = 8 + 20 * i;
            let cpu = u32_be(blob, e)?;
            let off = u32_be(blob, e + 8)? as usize;
            let size = u32_be(blob, e + 12)? as usize;
            Some((Some(cpu), blob.get(off..off.checked_add(size)?)?))
        })
        .collect()
}

/// The length-prefixed strings of a pipeline object's `__GPU_LD_MD` section (its function name
/// and stage).
fn ld_names(obj: &[u8]) -> Vec<&[u8]> {
    let Some(secs) = sections(obj) else {
        return vec![];
    };
    let Some(ld) = secs.iter().find(|s| s.seg == b"__GPU_LD_MD") else {
        return vec![];
    };
    let d = ld.data;
    let mut out = Vec::new();
    let mut o = 0;
    while o + 4 < d.len() {
        let n = u32_le(d, o).unwrap_or(0) as usize;
        if n > 0 && n < 256 && o + 4 + n < d.len() && d[o + 4 + n] == 0 {
            let w = &d[o + 4..o + 4 + n];
            if w.iter().all(|&c| (0x21..0x7f).contains(&c)) {
                out.push(w);
            }
        }
        o += 4;
    }
    out
}

/// The vtable position and size of the FlatBuffers table at `t` of `d`.
fn fb_vtable(d: &[u8], t: usize) -> Option<(usize, usize)> {
    let soff = i32::from_le_bytes(d.get(t..t.checked_add(4)?)?.try_into().ok()?);
    let vt = usize::try_from(t as i64 - soff as i64).ok()?;
    Some((vt, u16_le(d, vt)? as usize))
}

/// Field `field` of the FlatBuffers table at `t` of `d`: Ok(Some((position, u32 value))),
/// Ok(None) if the table has no such field (a default value), Err(()) if `t` is not a table.
#[allow(clippy::result_unit_err)]
fn fb_field(d: &[u8], t: usize, field: usize) -> Result<Option<(usize, u32)>, ()> {
    let (vt, vsz) = fb_vtable(d, t).ok_or(())?;
    if 4 + 2 * field + 2 > vsz {
        return Ok(None);
    }
    let fo = u16_le(d, vt + 4 + 2 * field).ok_or(())? as usize;
    if fo == 0 {
        return Ok(None);
    }
    Ok(Some((t + fo, u32_le(d, t + fo).ok_or(())?)))
}

/// The native per-thread stack of one pipeline object (None: metadata not understood; an
/// absent field is a zero stack, FlatBuffers omits default values).
fn object_stack(obj: &[u8]) -> Option<u32> {
    let secs = sections(obj)?;
    let meta = secs.iter().find(|s| s.sect == b"__compute")?.data;
    let root = u32_le(meta, 0)? as usize;
    let (pos, rel) = fb_field(meta, root, 0).ok()??;
    let table = pos.checked_add(rel as usize)?;
    Some(
        fb_field(meta, table, META_STACK_FIELD)
            .ok()?
            .map_or(0, |(_, v)| v),
    )
}

/// [NativeInfo] of the pipeline of function `name` in the binary archive `blob` (the contents of
/// a file zkp `archive` saved), None if it is not there or the layout is not understood.
pub(crate) fn native_info(blob: &[u8], name: &str) -> Option<NativeInfo> {
    let magic = MH_MAGIC_64.to_le_bytes();
    for (cpu, slice) in slices(blob) {
        if cpu.is_some_and(|c| c != CPU_APPLEGPU) {
            continue;
        }
        let Some(secs) = sections(slice) else {
            continue;
        };
        for comp in secs.iter().filter(|s| s.sect == b"__compute") {
            let d = comp.data;
            // The nested objects, one per pipeline: every Mach-O magic that parses.
            let starts: Vec<usize> = (0..d.len().saturating_sub(3))
                .filter(|&o| d[o..o + 4] == magic)
                .collect();
            let objs: Vec<usize> = starts
                .iter()
                .enumerate()
                .filter(|(i, &s)| {
                    let end = starts.get(i + 1).copied().unwrap_or(d.len());
                    sections(&d[s..end]).is_some()
                })
                .map(|(_, &s)| s)
                .collect();
            for (i, &s) in objs.iter().enumerate() {
                let obj = &d[s..objs.get(i + 1).copied().unwrap_or(d.len())];
                if ld_names(obj).first().copied() != Some(name.as_bytes()) {
                    continue;
                }
                let text = sections(obj)?
                    .iter()
                    .filter(|s| s.sect == b"__text")
                    .map(|s| s.data.len())
                    .sum();
                return Some(NativeInfo {
                    text,
                    stack: object_stack(obj)?,
                });
            }
        }
    }
    None
}

/// [native_info] of the archive file at `path`.
pub(crate) fn archive_info(path: &Path, name: &str) -> Option<NativeInfo> {
    native_info(&std::fs::read(path).ok()?, name)
}

// ---- The process-wide guard ----

/// The plain split kernel (the flat chain, R22).
pub(crate) const PLAIN_KERNEL: &str = "ef_eval_split";
/// The fallback candidates of R22 / R23, in order, each a different code shape for the back end:
/// the nested chain (the pre-R22 default, fast regime today) and R15e's form (16-byte row loads;
/// slow today, 36 KB stack). Since W7-R1 this is the order with [FALLBACK_ORDER_ENV]`=nest`.
pub(crate) const FALLBACK_KERNELS: [(&str, &str); 2] = [
    (super::eval_check::NEST_KERNEL, "nest"),
    ("ef_eval_split_r15e", "r15e"),
];

/// W7-R1 (R23 review P2): the default fallback order. R22's flat kernel without the hoisted trace
/// loads (`ef_eval_split_nohoist`, the HOIST=0 revert kernel: the closest code shape to the plain
/// kernel and only ~3 % slower than it, R23 3.3) first, then [FALLBACK_KERNELS]. The nested
/// chain is ~7 % slower than the flat one (R22), so it is the second choice.
pub(crate) const FALLBACK_KERNELS_NOHOIST_FIRST: [(&str, &str); 3] = [
    (super::eval_check::NOHOIST_KERNEL, "nohoist"),
    FALLBACK_KERNELS[0],
    FALLBACK_KERNELS[1],
];

/// W7-R1: `RISC0_METAL_RV32IM_GUARD_FALLBACK`: the guard's fallback order. Unset / `nohoist`
/// (default): [FALLBACK_KERNELS_NOHOIST_FIRST]; `nest`: [FALLBACK_KERNELS], the order before
/// W7-R1 (the revert knob). Read once, by the guard's thread; an unknown value is logged and the
/// default used.
pub(crate) const FALLBACK_ORDER_ENV: &str = "RISC0_METAL_RV32IM_GUARD_FALLBACK";

/// The fallback list [FALLBACK_ORDER_ENV]'s value `v` selects (None: unset).
pub(crate) fn parse_fallback_order(
    v: Option<&str>,
) -> anyhow::Result<&'static [(&'static str, &'static str)]> {
    match v.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("nohoist") | Some("1") => Ok(&FALLBACK_KERNELS_NOHOIST_FIRST),
        Some("nest") => Ok(&FALLBACK_KERNELS),
        Some(other) => {
            anyhow::bail!("{FALLBACK_ORDER_ENV}={other:?}: expected `nohoist` (default) or `nest`")
        }
    }
}

/// The guard's fallback candidates in order ([FALLBACK_ORDER_ENV]).
pub(crate) fn fallback_kernels() -> &'static [(&'static str, &'static str)] {
    let v = std::env::var(FALLBACK_ORDER_ENV).ok();
    parse_fallback_order(v.as_deref()).unwrap_or_else(|e| {
        tracing::warn!("{e:#}; using the default order");
        &FALLBACK_KERNELS_NOHOIST_FIRST
    })
}

/// The regime of the process library's plain `ef_eval_split`, read once from the archive the
/// library was loaded through.
pub(crate) fn plain_regime(kernels: &super::Rv32imKernels) -> Regime {
    static R: OnceLock<Regime> = OnceLock::new();
    *R.get_or_init(|| {
        let info = kernels
            .kernels
            .archive_report()
            .and_then(|r| r.path.as_deref())
            .and_then(|p| archive_info(p, PLAIN_KERNEL));
        Regime::of(info, settings().1)
    })
}

/// The fallback split kernel and PSO, set by [start] only when the plain one is slow and a
/// candidate is fast.
static FALLBACK: OnceLock<(&'static str, &'static metal::ComputePipelineState)> = OnceLock::new();

/// The split kernel and PSO to use instead of the plain `ef_eval_split` (None: the plain one).
pub(crate) fn fallback_split() -> Option<(&'static str, &'static metal::ComputePipelineState)> {
    FALLBACK.get().copied()
}

/// Whether the domain-specialized PSO of `name` in `fc` regime may be used: a slow one is
/// refused (the plain PSO is used for that domain) unless the plain kernel is slow too, or the
/// guard only warns. Logs the outcome.
pub(crate) fn fc_acceptable(
    kernels: &super::Rv32imKernels,
    name: &str,
    domain: u32,
    fc: Regime,
) -> bool {
    let (mode, _) = settings();
    if mode == GuardMode::Off {
        return true;
    }
    let Regime::Slow(_) = fc else {
        tracing::debug!(
            "rv32im Metal eval regime guard: {name} for domain {domain}: {}",
            fc.describe()
        );
        return true;
    };
    let plain = plain_regime(kernels);
    let refuse = mode == GuardMode::Fallback && !matches!(plain, Regime::Slow(_));
    tracing::warn!(
        "rv32im Metal eval regime guard: {name} for domain {domain}: {} (plain {PLAIN_KERNEL}: \
         {}); eval_check is ~2.4x slower in this regime (docs/metal-port/notes/R22.md); {}",
        fc.describe(),
        plain.describe(),
        if refuse {
            "using the plain PSO for this domain"
        } else {
            "using it anyway"
        }
    );
    !refuse
}

/// Once per process, on a background thread: check the plain split kernel's regime, warn if it
/// is slow, and (in [GuardMode::Fallback]) build the [fallback_kernels] through binary archives
/// of their own and use the first one in the fast regime.
pub(crate) fn start(kernels: &'static super::Rv32imKernels) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let (mode, max) = settings();
        if mode == GuardMode::Off || kernels.source.is_none() {
            return;
        }
        let spawned = std::thread::Builder::new()
            .name("rv32im-eval-guard".into())
            .spawn(move || {
                let t0 = std::time::Instant::now();
                let plain = plain_regime(kernels);
                let read = t0.elapsed();
                let force = mode == GuardMode::Force;
                match plain {
                    _ if force => tracing::warn!(
                        "rv32im Metal eval regime guard: {EVAL_GUARD_ENV}=force (testing): \
                         {PLAIN_KERNEL}: {}; switching to the first fallback that builds",
                        plain.describe()
                    ),
                    Regime::Fast(_) => {
                        tracing::info!(
                            "rv32im Metal eval regime guard: {PLAIN_KERNEL}: {} (limit {max} B; \
                             read in {read:.1?})",
                            plain.describe()
                        );
                        return;
                    }
                    Regime::Unknown => {
                        tracing::info!(
                            "rv32im Metal eval regime guard: {PLAIN_KERNEL}: {}",
                            plain.describe()
                        );
                        return;
                    }
                    Regime::Slow(_) => tracing::warn!(
                        "rv32im Metal eval regime guard: {PLAIN_KERNEL}: {} (limit {max} B). The \
                         Metal compiler put the eval kernel in its slow regime (R18/R21: ~2.4x \
                         eval_check time, ~+45 % per segment). {}",
                        plain.describe(),
                        if mode == GuardMode::Warn {
                            format!("{EVAL_GUARD_ENV}=warn: no fallback.")
                        } else {
                            "Trying the fallback kernels in the background.".into()
                        }
                    ),
                }
                if mode == GuardMode::Warn {
                    return;
                }
                let order = fallback_kernels();
                tracing::info!(
                    "rv32im Metal eval regime guard: fallback order {:?} ({FALLBACK_ORDER_ENV})",
                    order.iter().map(|k| k.0).collect::<Vec<_>>()
                );
                for &(name, tag) in order {
                    let res =
                        metal::objc::rc::autoreleasepool(|| build_fallback(kernels, name, tag));
                    match res {
                        Ok((pso, r)) if force || matches!(r, Regime::Fast(_)) => {
                            tracing::warn!(
                                "rv32im Metal eval regime guard: using {name} instead of \
                                 {PLAIN_KERNEL} ({}{}; {EVAL_GUARD_ENV}=warn keeps \
                                 {PLAIN_KERNEL})",
                                r.describe(),
                                if force { ", force" } else { "" }
                            );
                            let _ = FALLBACK.set((name, Box::leak(Box::new(pso))));
                            return;
                        }
                        Ok((_, r)) => tracing::warn!(
                            "rv32im Metal eval regime guard: fallback {name}: {}",
                            r.describe()
                        ),
                        Err(e) => tracing::warn!(
                            "rv32im Metal eval regime guard: cannot build {name} ({e:#})"
                        ),
                    }
                }
                tracing::warn!(
                    "rv32im Metal eval regime guard: no fallback in the fast regime; keeping \
                     {PLAIN_KERNEL}"
                );
            });
        if let Err(e) = spawned {
            tracing::warn!("rv32im Metal: cannot start the eval regime guard: {e}");
        }
    });
}

/// Build the fallback `name` through a binary archive of its own (unit `<unit>_<tag>`) and read
/// its regime from that archive.
fn build_fallback(
    kernels: &super::Rv32imKernels,
    name: &str,
    tag: &str,
) -> anyhow::Result<(metal::ComputePipelineState, Regime)> {
    let src = kernels
        .source
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no kernel source"))?;
    let lib = kernels.kernels.library();
    let (pso, r) = risc0_zkp::hal::metal::archive::specialized_pipeline(
        src.archive_dir.clone(),
        lib.device(),
        lib,
        src.bytes,
        &format!("{}_{tag}", src.unit),
        name,
        &[],
    )?;
    let info = r.path.as_deref().and_then(|p| archive_info(p, name));
    Ok((pso, Regime::of(info, settings().1)))
}

#[cfg(test)]
mod tests {
    //! Parser and mode tests; the kernels' regimes are checked against real archives in
    //! eval_check.rs (`ef_tests::eval_regime_guard_kernels`). Nothing here reads or sets the
    //! guard's environment variables (PLAN 8.7).
    use super::*;

    #[test]
    fn guard_mode_parsing() {
        assert_eq!(GuardMode::parse("").unwrap(), GuardMode::Fallback);
        assert_eq!(GuardMode::parse("1").unwrap(), GuardMode::Fallback);
        assert_eq!(GuardMode::parse(" Warn ").unwrap(), GuardMode::Warn);
        assert_eq!(GuardMode::parse("0").unwrap(), GuardMode::Off);
        assert_eq!(GuardMode::parse("off").unwrap(), GuardMode::Off);
        assert!(GuardMode::parse("2").is_err());
        let i = NativeInfo {
            text: 1,
            stack: STACK_MAX,
        };
        assert_eq!(Regime::of(Some(i), STACK_MAX), Regime::Fast(i));
        let j = NativeInfo {
            text: 1,
            stack: STACK_MAX + 1,
        };
        assert_eq!(Regime::of(Some(j), STACK_MAX), Regime::Slow(j));
        assert_eq!(Regime::of(None, STACK_MAX), Regime::Unknown);
    }

    #[test]
    fn fallback_order_parsing() {
        let names = |v: Option<&str>| -> Vec<&str> {
            parse_fallback_order(v)
                .unwrap()
                .iter()
                .map(|k| k.0)
                .collect()
        };
        let new = vec![
            "ef_eval_split_nohoist",
            "ef_eval_split_nest",
            "ef_eval_split_r15e",
        ];
        assert_eq!(names(None), new);
        assert_eq!(names(Some("")), new);
        assert_eq!(names(Some(" NoHoist ")), new);
        assert_eq!(
            names(Some("nest")),
            vec!["ef_eval_split_nest", "ef_eval_split_r15e"]
        );
        assert!(parse_fallback_order(Some("r15e")).is_err());
        // Every candidate has its own archive unit tag.
        let tags: std::collections::HashSet<_> =
            FALLBACK_KERNELS_NOHOIST_FIRST.iter().map(|k| k.1).collect();
        assert_eq!(tags.len(), FALLBACK_KERNELS_NOHOIST_FIRST.len());
    }

    #[test]
    fn parser_survives_garbage() {
        assert_eq!(native_info(&[], "x"), None);
        assert_eq!(native_info(&[0xcb, 0xfe, 0xba, 0xbe], "x"), None);
        // A fat header whose slice points past the end, and a slice count that is absurd.
        let mut fat = vec![0xcb, 0xfe, 0xba, 0xbe, 0, 0, 0, 1];
        fat.extend_from_slice(&[
            1, 0, 0, 0x13, 0, 0, 0, 0, 0, 0, 0, 0x30, 0xff, 0xff, 0xff, 0xff,
        ]);
        fat.extend_from_slice(&[0, 0, 0, 4]);
        assert_eq!(native_info(&fat, "x"), None);
        let mut many = vec![0xca, 0xfe, 0xba, 0xbe, 0xff, 0xff, 0xff, 0xff];
        many.resize(64, 0);
        assert_eq!(native_info(&many, "x"), None);
        // Mach-O magics with junk behind them, inside and outside a fat image.
        let mut junk = Vec::new();
        for i in 0..4096u32 {
            junk.extend_from_slice(&(i.wrapping_mul(2654435761)).to_le_bytes());
            if i % 97 == 0 {
                junk.extend_from_slice(&MH_MAGIC_64.to_le_bytes());
            }
        }
        assert_eq!(native_info(&junk, "x"), None);
        let mut thin = MH_MAGIC_64.to_le_bytes().to_vec();
        thin.extend_from_slice(&junk);
        assert_eq!(native_info(&thin, "x"), None);
    }
}
