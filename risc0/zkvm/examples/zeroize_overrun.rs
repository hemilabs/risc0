// Deterministic test for the eltwise_zeroize out-of-bounds over-run.
//
// eltwise_zeroize_fp has (had) no bounds check, while launchKernel rounds the
// grid up to a multiple of the block size. So zeroizing a buffer whose length
// isn't a block-size multiple over-reads AND over-writes the trailing threads
// past the end of the buffer, into adjacent memory.
//
// Test: fill a 256-elem buffer with INVALID (0xffffffff), zeroize only the
// first 90 (like the `global` buffer, REGCOUNT_GLOBAL=90), then inspect
// elements [90,256). If the kernel is bounds-checked they stay INVALID; if it
// over-runs they get zeroized (INVALID -> 0).

use risc0_core::field::baby_bear::BabyBearElem as Val;
use risc0_zkp::hal::{Buffer, Hal};

fn main() {
    #[cfg(feature = "rocm")]
    let hal = risc0_zkp::hal::hip::HipHalSha256::new();
    #[cfg(feature = "cuda")]
    let hal = risc0_zkp::hal::cuda::CudaHalSha256::new();

    const N: usize = 256;
    const SUB: usize = 90;
    const INVALID: u32 = 0xffff_ffff;

    let buf = hal.alloc_elem_init("overrun_test", N, Val::new_raw(INVALID));
    // Zeroize only the first SUB elements (count == SUB).
    let sub = buf.slice(0, SUB);
    hal.eltwise_zeroize_elem(&sub);

    let out = buf.to_vec();
    let raw: Vec<u32> = out.iter().map(|e| e.as_u32_montgomery()).collect();

    // Sanity: first SUB should be zeroized (INVALID -> 0).
    let zeroized_front = (0..SUB).all(|i| raw[i] == 0);
    // The over-run region [SUB, N): expect still INVALID if bounds-checked.
    let clobbered: Vec<usize> = (SUB..N).filter(|&i| raw[i] != INVALID).collect();

    println!("front[0..{SUB}] all zeroized: {zeroized_front}");
    if clobbered.is_empty() {
        println!("PASS: no over-run — elements [{SUB},{N}) untouched (still INVALID)");
    } else {
        println!(
            "FAIL: OVER-RUN — {} of {} trailing elements clobbered (idx {}..={}), e.g. raw[{}]={:#010x}",
            clobbered.len(),
            N - SUB,
            clobbered[0],
            clobbered[clobbered.len() - 1],
            clobbered[0],
            raw[clobbered[0]],
        );
    }
}
