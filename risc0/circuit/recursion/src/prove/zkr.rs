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

use std::{
    collections::HashMap,
    io::Read as _,
    sync::{Arc, Mutex, MutexGuard, OnceLock},
};

use anyhow::{anyhow, bail, Result};

use super::Program;

/// The program `name` for `po2` cycles.
///
/// Shared: every lift and join asks for its program, and building one from its words
/// ([Program::from_encoded]: 12-24 MB, 1.1-2.2 ms on an M5 Max) plus the prover's re-check of a
/// new `Arc`'s code against its ctrl caches (a 16-24 MB compare; pointer equality skips it)
/// cost 1.1-2.0 ms per Metal lift and 2.0-2.6 ms per join (1-2 %, notes/P31-10.md). So the built programs
/// are kept per process, keyed by (name, po2) (P31-10, as fork 3.0's F30 cache), and every call
/// for the same program returns the same `Arc`. Like stock's word memo, a zkr is decoded at most
/// once per process: no name is ever evicted. Revert knob: `RISC0_METAL_REC_ZKR_CACHE=0`
/// (stock 3.1: the decoded words are kept per name, a new [Program] is built on every call).
pub fn get_zkr(name: &str, po2: usize) -> Result<Arc<Program>> {
    if !program_cache_enabled() {
        let encoded = cached_zkr(name)?;
        return Ok(Arc::new(Program::from_encoded(&encoded, po2)));
    }
    program_cache::get(name, po2)
}

/// `RISC0_METAL_REC_ZKR_CACHE`: `0` (or `off`/`false`) turns the [get_zkr] program cache off.
const ZKR_CACHE_ENV: &str = "RISC0_METAL_REC_ZKR_CACHE";

fn program_cache_enabled() -> bool {
    #[cfg(test)]
    if let Some(on) = program_cache::test_override() {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var(ZKR_CACHE_ENV).as_deref().map(str::trim),
            Ok("0" | "off" | "false")
        )
    })
}

/// The [get_zkr] program cache: per zkr name, the programs built for its po2s (at most
/// [program_cache::PO2_CAP], LRU among them). It replaces the per-name word memo ([cached_zkr],
/// not filled while the cache is on), with the same bound: one copy of each zkr the process has
/// used, never decoded twice. Names are never evicted (fix round 1): an LRU over names made every
/// reuse of an evicted zkr pay the 40-85 ms xz decode again, where stock 3.1 pays ~1.2 ms.
/// Every zkvm call uses `RECURSION_PO2`, so in practice each name holds one program: the memory
/// stock 3.1's memo holds as words (all 32 zkrs: ~470 MB in either; a prover uses 4-13).
mod program_cache {
    use super::*;

    /// Programs kept per name: a second po2 of a name is built from the first (a code copy, no
    /// decode). Only tests ask for more than one po2; evicting among a name's po2s always keeps
    /// one, so the zkr is never decoded again.
    pub(super) const PO2_CAP: usize = 2;

    struct Entry {
        program: Arc<Program>,
        last_use: u64,
    }

    #[derive(Default)]
    struct Cache {
        by_name: HashMap<String, Vec<Entry>>,
        clock: u64,
    }

    fn cache() -> MutexGuard<'static, Cache> {
        static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
        CACHE
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// `program` for `po2` (its code does not depend on the po2; same check as
    /// [Program::from_encoded]).
    fn with_po2(program: &Program, po2: usize) -> Program {
        let prog = Program {
            code: program.code.clone(),
            code_size: program.code_size,
            po2,
        };
        assert!(prog.code.len() <= (prog.code_size * ((1 << po2) - risc0_zkp::ZK_CYCLES)));
        prog
    }

    pub(super) fn get(name: &str, po2: usize) -> Result<Arc<Program>> {
        // Another po2 of the same name: built from its code, without decoding the zkr again.
        let other = {
            let mut c = cache();
            c.clock += 1;
            let now = c.clock;
            match c.by_name.get_mut(name) {
                Some(entries) => {
                    if let Some(e) = entries.iter_mut().find(|e| e.program.po2 == po2) {
                        e.last_use = now;
                        return Ok(e.program.clone());
                    }
                    entries
                        .iter()
                        .max_by_key(|e| e.last_use)
                        .map(|e| e.program.clone())
                }
                None => None,
            }
        };
        // Built outside the lock: two threads may both build a program the first time.
        let program = Arc::new(match other {
            Some(other) => with_po2(&other, po2),
            None => {
                #[cfg(test)]
                note_decode(name);
                Program::from_encoded(&extract_zkr(ZkrSection::for_name(name), name)?, po2)
            }
        });
        let mut c = cache();
        c.clock += 1;
        let now = c.clock;
        let entries = c.by_name.entry(name.to_string()).or_default();
        if let Some(e) = entries.iter_mut().find(|e| e.program.po2 == po2) {
            e.last_use = now;
            return Ok(e.program.clone());
        }
        if entries.len() >= PO2_CAP {
            let oldest = (0..entries.len())
                .min_by_key(|&i| entries[i].last_use)
                .unwrap();
            entries.swap_remove(oldest);
        }
        entries.push(Entry {
            program: program.clone(),
            last_use: now,
        });
        Ok(program)
    }

    #[cfg(test)]
    static DECODES: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

    #[cfg(test)]
    fn note_decode(name: &str) {
        *DECODES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(Default::default)
            .entry(name.to_string())
            .or_default() += 1;
    }

    /// xz decodes of `name` by the cache so far (tests).
    #[cfg(test)]
    pub(super) fn decodes(name: &str) -> usize {
        DECODES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .and_then(|m| m.get(name).copied())
            .unwrap_or(0)
    }

    /// Programs cached for `name` (tests).
    #[cfg(test)]
    pub(super) fn len(name: &str) -> usize {
        cache().by_name.get(name).map_or(0, Vec::len)
    }

    /// Bytes of code the cache holds (tests).
    #[cfg(test)]
    pub(super) fn bytes() -> usize {
        cache()
            .by_name
            .values()
            .flatten()
            .map(|e| e.program.code.len() * std::mem::size_of::<u32>())
            .sum()
    }

    #[cfg(test)]
    static OVERRIDE: Mutex<Option<bool>> = Mutex::new(None);

    #[cfg(test)]
    pub(super) fn test_override() -> Option<bool> {
        *OVERRIDE.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Force the cache on/off (`None`: the env) in this process (tests; hold
    /// `tests::override_lock`, the crate's global-state test lock under Metal: it is process-wide).
    #[cfg(test)]
    pub(crate) fn set_test_override(on: Option<bool>) {
        *OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) = on;
    }
}

/// The decoded zkr, decoded once per process: every lift and join asks for its program, and
/// decompressing one (15-24 MB of xz) takes 50-80 ms, about twice what an RTX 5090 takes to prove
/// it. A prover uses a handful of them (its segments' lifts, join, identity), tens of MB in all.
fn cached_zkr(name: &str) -> Result<Arc<Vec<u32>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<Vec<u32>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(zkr) = cache.lock().unwrap().get(name) {
        return Ok(zkr.clone());
    }
    // Decoded outside the lock: two threads may both decode a zkr the first time, not wait.
    let zkr = Arc::new(extract_zkr(ZkrSection::for_name(name), name)?);
    Ok(cache
        .lock()
        .unwrap()
        .entry(name.to_string())
        .or_insert(zkr)
        .clone())
}

/// Iterate over all provided zkr programs, sorted by name.
///
/// ```rust
/// let listing = risc0_circuit_recursion::prove::zkr::get_all_zkrs().unwrap();
/// println!("{}", listing.into_iter().map(|(name, _)| name).collect::<Vec<_>>().join("\n"));
/// ```
pub fn get_all_zkrs() -> Result<Vec<(String, Vec<u32>)>> {
    sorted_names()
        .into_iter()
        .map(|(section, name)| Ok((name.into(), extract_zkr(section, name)?)))
        .collect()
}

/// The order matters: `cargo xtask bootstrap` builds the allowed control root
/// from [get_all_zkrs]. Sort to match the zip these zkrs used to ship in.
fn sorted_names() -> Vec<(ZkrSection, &'static str)> {
    let mut names: Vec<_> = [ZkrSection::Lift, ZkrSection::Control]
        .into_iter()
        .flat_map(|section| section.table().keys().map(move |&name| (section, name)))
        .collect();
    names.sort_by_key(|&(_, name)| name);
    names
}

/// The zkrs are split across two crates to stay under the crates.io size
/// limit: the lifts in one, everything else (join, resolve, etc.) in the other.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ZkrSection {
    Lift,
    Control,
}

impl ZkrSection {
    fn for_name(name: &str) -> Self {
        if name.starts_with("lift_") {
            Self::Lift
        } else {
            Self::Control
        }
    }

    fn table(&self) -> &'static phf::Map<&'static str, (&'static [u8], usize)> {
        match self {
            Self::Lift => &risc0_circuit_recursion_zkrs::ZKRS,
            Self::Control => &risc0_circuit_recursion_control_zkrs::ZKRS,
        }
    }
}

fn extract_zkr(section: ZkrSection, name: &str) -> Result<Vec<u32>> {
    let &(xz_data, uncompressed_size) = section
        .table()
        .get(name)
        .ok_or_else(|| anyhow!("given zkr not found: {name}"))?;

    if !uncompressed_size.is_multiple_of(std::mem::size_of::<u32>()) {
        bail!(".zkr is incorrect size");
    }

    let mut decoder = liblzma::read::XzDecoder::new_parallel(xz_data);
    let mut u32s = vec![0u32; uncompressed_size / std::mem::size_of::<u32>()];
    decoder.read_exact(bytemuck::cast_slice_mut(&mut u32s))?;
    Ok(u32s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_sorted_and_in_right_section() {
        let names = sorted_names();
        assert_eq!(names.len(), 32);
        assert!(names.is_sorted_by_key(|&(_, name)| name));
        for (section, name) in names {
            assert_eq!(section, ZkrSection::for_name(name), "{name}");
        }
    }

    #[test]
    fn get_zkr_from_each_section() {
        get_zkr("lift_rv32im_v2_14.zkr", 18).unwrap();
        get_zkr("test_recursion_circuit.zkr", 18).unwrap();
        assert!(get_zkr("missing.zkr", 18).is_err());
    }

    /// Serializes the tests that set the process-wide cache override (PLAN 8.7): the crate's
    /// global-state test lock under Metal (the Metal tests that rely on shared program `Arc`s
    /// take it too), a local one otherwise.
    fn override_lock() -> MutexGuard<'static, ()> {
        #[cfg(feature = "metal")]
        return crate::prove::hal::metal::global_state_test_lock();
        #[cfg(not(feature = "metal"))]
        {
            static LOCK: Mutex<()> = Mutex::new(());
            LOCK.lock().unwrap_or_else(|e| e.into_inner())
        }
    }

    fn same_program(a: &Program, b: &Program) -> bool {
        a.code == b.code && a.code_size == b.code_size && a.po2 == b.po2
    }

    #[test]
    fn p31_10_cache_shares_and_matches_stock() {
        let _lock = override_lock();
        let name = "lift_rv32im_v2_14.zkr";
        let stock =
            Program::from_encoded(&extract_zkr(ZkrSection::for_name(name), name).unwrap(), 18);
        program_cache::set_test_override(Some(true));
        let a = get_zkr(name, 18).unwrap();
        let b = get_zkr(name, 18).unwrap();
        // Another po2 of a cached program: built from its words, same code.
        let c = get_zkr(name, 19).unwrap();
        let d = get_zkr(name, 19).unwrap();
        program_cache::set_test_override(Some(false));
        let off1 = get_zkr(name, 18).unwrap();
        let off2 = get_zkr(name, 18).unwrap();
        program_cache::set_test_override(None);
        assert!(
            Arc::ptr_eq(&a, &b),
            "the cache hands out one Arc per (name, po2)"
        );
        assert!(Arc::ptr_eq(&c, &d));
        assert!(!Arc::ptr_eq(&a, &c));
        assert!(
            same_program(&a, &stock),
            "cached program == stock from_encoded"
        );
        assert_eq!(c.po2, 19);
        assert_eq!(c.code, stock.code);
        assert!(
            !Arc::ptr_eq(&off1, &off2),
            "knob off: a new program per call (stock 3.1)"
        );
        assert!(same_program(&off1, &stock) && same_program(&off2, &stock));
        assert!(!Arc::ptr_eq(&off1, &a));
    }

    #[test]
    fn p31_10_po2_variants_bounded_without_decode() {
        let _lock = override_lock();
        program_cache::set_test_override(Some(true));
        let name = "lift_rv32im_v2_15.zkr";
        let first = get_zkr(name, 22).unwrap();
        let decodes = program_cache::decodes(name);
        let mut got = Vec::new();
        for po2 in [20, 21, 22, 20, 21, 22, 19] {
            got.push(get_zkr(name, po2).unwrap());
            assert!(program_cache::len(name) <= program_cache::PO2_CAP);
        }
        let again = get_zkr(name, 19).unwrap();
        let bytes = program_cache::bytes();
        program_cache::set_test_override(None);
        assert_eq!(
            program_cache::decodes(name),
            decodes,
            "a po2 variant re-ran the xz decode"
        );
        assert!(Arc::ptr_eq(got.last().unwrap(), &again));
        println!("P3110 cache: {bytes} B of code held");
        let words = extract_zkr(ZkrSection::Lift, name).unwrap();
        for p in got.iter().chain([&first]) {
            assert!(
                same_program(p, &Program::from_encoded(&words, p.po2)),
                "po2 {}",
                p.po2
            );
        }
    }

    /// Review finding (fix round 1): a process that proves with more distinct programs than the
    /// cache once held (CAP = 8 with LRU eviction) re-ran the 40-85 ms xz decode on every reuse
    /// (failed on c160a39a0: decodes 1 -> 2 for each of the 10). Cycle 10 lifts twice: the second
    /// round must not decode any of them again and must hand out the same `Arc`s.
    #[test]
    fn p31_10_no_xz_decode_after_many_programs() {
        let _lock = override_lock();
        program_cache::set_test_override(Some(true));
        let names: Vec<String> = (14..24)
            .map(|po2| format!("lift_rv32im_v2_{po2}.zkr"))
            .collect();
        let first: Vec<_> = names.iter().map(|n| get_zkr(n, 22).unwrap()).collect();
        let before: Vec<usize> = names.iter().map(|n| program_cache::decodes(n)).collect();
        let second: Vec<_> = names.iter().map(|n| get_zkr(n, 22).unwrap()).collect();
        let after: Vec<usize> = names.iter().map(|n| program_cache::decodes(n)).collect();
        program_cache::set_test_override(None);
        assert_eq!(before, after, "steady-state get_zkr re-ran the xz decode");
        assert!(first.iter().zip(&second).all(|(a, b)| Arc::ptr_eq(a, b)));
    }

    /// P31-10 profile (ignored; run by hand, release): wall time of `get_zkr` for the programs
    /// a prover asks for on every lift/join, first call (xz decode) and steady state, and the
    /// parts of the steady state. One `P3110ZKR ...` line per program.
    #[test]
    #[ignore]
    fn p31_10_profile_get_zkr() {
        use std::time::Instant;
        let reps: usize = std::env::var("P3110_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(15);
        let _lock = override_lock();
        let cache_on = std::env::var(ZKR_CACHE_ENV).as_deref() != Ok("0");
        program_cache::set_test_override(Some(cache_on));
        for name in [
            "lift_rv32im_v2_20.zkr",
            "lift_rv32im_v2_21.zkr",
            "lift_rv32im_v2_18.zkr",
            "join.zkr",
            "resolve.zkr",
            "identity.zkr",
        ] {
            let t = Instant::now();
            let p = get_zkr(name, 18).unwrap();
            let first = t.elapsed().as_secs_f64() * 1e3;
            drop(p);
            let mut times = Vec::new();
            let mut conv = Vec::new();
            let mut drops = Vec::new();
            let encoded = extract_zkr(ZkrSection::for_name(name), name).unwrap();
            for _ in 0..reps {
                let t = Instant::now();
                let p = get_zkr(name, 18).unwrap();
                times.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                drop(p);
                drops.push(t.elapsed().as_secs_f64() * 1e3);
                let t = Instant::now();
                let p = Program::from_encoded(&encoded, 18);
                conv.push(t.elapsed().as_secs_f64() * 1e3);
                std::hint::black_box(&p);
            }
            let med = |v: &mut Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (v[v.len() / 2], v[0], v[v.len() - 1])
            };
            let (m, lo, hi) = med(&mut times);
            let (cm, _, _) = med(&mut conv);
            let (dm, _, _) = med(&mut drops);
            println!(
                "P3110ZKR cache={cache_on} {name} words={} bytes={} first={first:.2}ms steady_median={m:.2}ms \
                 min={lo:.2} max={hi:.2} from_encoded={cm:.2}ms drop={dm:.2}ms n={reps}",
                encoded.len(),
                encoded.len() * 4,
            );
        }
        program_cache::set_test_override(None);
    }
}
