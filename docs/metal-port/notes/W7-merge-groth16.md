# W7 groth16 unit merge record (2026-11-15)

Main checkout `zkminer/metal` at 2cc7525b2 before and after this unit: nothing was merged.

## Not merged: `metal/W7-G1` (7a7098de1, status unresolved)

The branch is a single commit on top of `zkminer/metal` (ancestry checked, no WIP commits). Both remaining
blocking findings describe the same defect. It is a real defect, not a false positive and not an unproven
speed claim, so the item is held back.

1. `risc0/groth16/src/prove/metal_witness.rs`: `RESERVED` is one shared `(count, latest timestamp)` pair.
   `reserve_for_wrap` increments the count and refreshes the timestamp. `WrapGuard::begin` decrements it.
   Nothing removes an expired reservation (only the test-only `clear_reservations` does). One prestart whose
   wrap never comes leaves count = 1. Every later prestart refreshes the timestamp, so `reserved_locked()`
   stays true at each wrap's after-witness `release_slot`. The graph release (and `release_idle_now`) is then
   disabled for the rest of the process, silently. The note's claim that a leaked reservation "costs at most
   30 s of a resident graph" is false.
2. The reviewer reproduced it (`scratch/w7/W7-G1-review3/leak.log`, 8 GiB budget, `MallocLargeCache=0`,
   `kind=ParTransient`). Without the extra prestart the graph was released after every wrap (footprint
   1372 / 2661 / 2661 MiB). With one prestart and no wrap, it stayed cached after every wrap (2326 / 3615 /
   3615 MiB). That is about +954 MiB held for good, which is the saving the item exists to deliver. The graph
   that stays resident is the `par_transient` one, which G15 rejected for raising the STARK-phase peak.

Fix needed before merging: expire each reservation on its own, e.g. a `VecDeque<Instant>` under `SLOT`.
Prune entries older than `reserve_ttl()` in `reserved_locked`/`reserve_for_wrap`, and have `WrapGuard::begin`
pop the oldest live one. Also add a kept test: leak one reservation, wait past a short
`RISC0_G16_GRAPH_RESERVE_MS`, run prestart + wrap, and assert `!is_cached()`. Then correct the note.
The worktree `/Users/user/zkminer/deps/risc0-wt-W7-G1` and branch `metal/W7-G1` are kept for that fix round.

## Gate

Not run: nothing was merged, so the main checkout is unchanged from the recursion unit's gated head
(2cc7525b2, see `W7-merge-recursion.md`).
