//! P31-14: verify receipts with the unmodified crates.io risc0-zkvm 3.1.0 verifier.
//!
//! ```text
//! stock-verify31 verify <receipt file> <image id hex> [--expect-journal <hex> | --expect-journal-file <path>] [--relabel-as-31]
//! stock-verify31 consts
//! ```
//!
//! `--relabel-as-31` (negative tests of receipts made by another version) first overwrites every
//! version label the receipt carries with the stock 3.1 value: the rv32im seal version word (3),
//! and the verifier-parameter digests of each segment, composite, succinct (also assumptions) and
//! Groth16 receipt. A 3.0 receipt is rejected by those label checks first; relabelled, it must
//! still be rejected by the cryptographic checks (control root, FRI, Groth16 pairing).
//!
//! A receipt file is a `risc0_zkvm::Receipt` serialized with bincode 1.3, the encoding the 3.1
//! crate itself uses for receipts (its `api` client/server and `Asset::Inline`). `verify` prints
//! what the receipt carries (kind, seal versions, control ids, verifier parameter digests), then
//! runs the stock `Receipt::verify(image_id)` (default `VerifierContext`: the 3.1 control root,
//! rv32im seal v3, Groth16 3.1 key) and, if asked, compares the journal bytes. Exit code 0 means
//! verified; 1 means rejected (the error is printed); 2 is a usage or I/O error.

use std::process::ExitCode;

use anyhow::{bail, Context, Result};
use risc0_zkvm::{
    sha::{Digest, Digestible},
    CompositeReceiptVerifierParameters, Groth16ReceiptVerifierParameters, InnerAssumptionReceipt,
    InnerReceipt, Receipt, SegmentReceiptVerifierParameters, SuccinctReceiptVerifierParameters,
    ALLOWED_CONTROL_ROOT,
};

fn usage() -> ExitCode {
    eprintln!(
        "usage: stock-verify31 verify <receipt file> <image id hex> [--expect-journal <hex> | \
         --expect-journal-file <path>] [--relabel-as-31]\n       stock-verify31 consts"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("consts") => {
            consts();
            ExitCode::SUCCESS
        }
        Some("verify") if args.len() >= 3 => {
            let mut rest: Vec<String> = args[3..].to_vec();
            let relabel = rest.iter().any(|a| a == "--relabel-as-31");
            rest.retain(|a| a != "--relabel-as-31");
            let expect = match parse_expect(&rest) {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("error: {e:#}");
                    return usage();
                }
            };
            let (mut receipt, image_id) = match load(&args[1], &args[2]) {
                Ok(x) => x,
                Err(e) => {
                    eprintln!("error: {e:#}");
                    return ExitCode::from(2);
                }
            };
            if relabel {
                relabel_as_31(&mut receipt.inner);
                receipt.metadata.verifier_parameters = receipt.inner.verifier_parameters();
                println!("relabelled: every version label set to the stock 3.1 value");
            }
            describe(&receipt);
            match check(&receipt, image_id, expect.as_deref()) {
                Ok(()) => {
                    println!("STOCK 3.1 VERIFY: OK ({})", args[1]);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    println!("STOCK 3.1 VERIFY: REJECTED ({}): {e:#}", args[1]);
                    ExitCode::from(1)
                }
            }
        }
        _ => usage(),
    }
}

fn parse_expect(rest: &[String]) -> Result<Option<Vec<u8>>> {
    match rest {
        [] => Ok(None),
        [flag, v] if flag == "--expect-journal" => Ok(Some(hex::decode(v)?)),
        [flag, v] if flag == "--expect-journal-file" => {
            Ok(Some(std::fs::read(v).with_context(|| format!("read {v}"))?))
        }
        _ => bail!("unknown arguments {rest:?}"),
    }
}

fn load(path: &str, image_id: &str) -> Result<(Receipt, Digest)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {path}"))?;
    let receipt: Receipt = bincode::deserialize(&bytes)
        .with_context(|| format!("{path} is not a bincode risc0-zkvm 3.1 Receipt"))?;
    let image_id = Digest::try_from(hex::decode(image_id.trim_start_matches("0x"))?.as_slice())
        .map_err(|_| anyhow::anyhow!("image id must be 32 bytes of hex"))?;
    Ok((receipt, image_id))
}

fn check(receipt: &Receipt, image_id: Digest, expect: Option<&[u8]>) -> Result<()> {
    receipt.verify(image_id)?;
    if let Some(j) = expect {
        if receipt.journal.bytes != j {
            bail!(
                "journal differs from the expected one ({} vs {} bytes)",
                receipt.journal.bytes.len(),
                j.len()
            );
        }
        println!("journal == expected ({} bytes)", j.len());
    }
    Ok(())
}

/// The 3.0 `SuccinctReceiptVerifierParameters::default()` digest (control root a54dc85a...).
fn succinct_30() -> Digest {
    Digest::try_from(
        hex::decode("ece5e9b8ae2cd6ea6b1827b464ff0348f9a7f4decd269c0087fdfd75098da013")
            .unwrap()
            .as_slice(),
    )
    .unwrap()
}

fn relabel_composite(c: &mut risc0_zkvm::CompositeReceipt) {
    let seg = SegmentReceiptVerifierParameters::default().digest();
    for s in &mut c.segments {
        if let Some(v) = s.seal.first_mut() {
            *v = risc0_circuit_rv32im::RV32IM_SEAL_VERSION;
        }
        s.verifier_parameters = seg;
    }
    for a in &mut c.assumption_receipts {
        match a {
            InnerAssumptionReceipt::Composite(c) => relabel_composite(c),
            // Only the recursion succinct parameters changed in 3.1; a keccak assumption carries
            // the keccak circuit's own parameters, the same in 3.0 and 3.1, so it is left alone.
            InnerAssumptionReceipt::Succinct(r) if r.verifier_parameters == succinct_30() => {
                r.verifier_parameters = SuccinctReceiptVerifierParameters::default().digest()
            }
            InnerAssumptionReceipt::Groth16(g) => {
                g.verifier_parameters = Groth16ReceiptVerifierParameters::default().digest()
            }
            _ => {}
        }
    }
    c.verifier_parameters = CompositeReceiptVerifierParameters::default().digest();
}

fn relabel_as_31(inner: &mut InnerReceipt) {
    match inner {
        InnerReceipt::Composite(c) => relabel_composite(c),
        InnerReceipt::Succinct(s) => {
            s.verifier_parameters = SuccinctReceiptVerifierParameters::default().digest()
        }
        InnerReceipt::Groth16(g) => {
            g.verifier_parameters = Groth16ReceiptVerifierParameters::default().digest()
        }
        _ => {}
    }
}

fn short(d: &Digest) -> String {
    hex::encode(d.as_bytes())
}

fn describe(receipt: &Receipt) {
    let claim_digest = receipt
        .claim()
        .map(|c| short(&c.digest()))
        .unwrap_or_else(|e| format!("<{e}>"));
    println!(
        "journal: {} bytes, claim digest {claim_digest}",
        receipt.journal.bytes.len()
    );
    match &receipt.inner {
        InnerReceipt::Composite(c) => {
            println!(
                "kind: composite, {} segments, {} assumption receipts",
                c.segments.len(),
                c.assumption_receipts.len()
            );
            for s in &c.segments {
                println!(
                    "  segment {}: seal version (word 0) {}, {} words, hashfn {}, verifier params {}",
                    s.index,
                    s.seal.first().copied().unwrap_or(u32::MAX),
                    s.seal.len(),
                    s.hashfn,
                    short(&s.verifier_parameters)
                );
            }
            for (i, a) in c.assumption_receipts.iter().enumerate() {
                match a {
                    InnerAssumptionReceipt::Succinct(r) => println!(
                        "  assumption {i}: succinct, hashfn {}, control id {}, verifier params {}",
                        r.hashfn,
                        short(&r.control_id),
                        short(&r.verifier_parameters)
                    ),
                    other => println!("  assumption {i}: {other:?}"),
                }
            }
        }
        InnerReceipt::Succinct(s) => println!(
            "kind: succinct, {} seal words, hashfn {}, control id {}, verifier params {}",
            s.seal.len(),
            s.hashfn,
            short(&s.control_id),
            short(&s.verifier_parameters)
        ),
        InnerReceipt::Groth16(g) => println!(
            "kind: groth16, {} seal bytes, verifier params {} (selector {})",
            g.seal.len(),
            short(&g.verifier_parameters),
            hex::encode(&g.verifier_parameters.as_bytes()[..4])
        ),
        InnerReceipt::Fake(_) => println!("kind: fake"),
        _ => println!("kind: unknown"),
    }
}

fn consts() {
    let g16 = Groth16ReceiptVerifierParameters::default();
    let seg = SegmentReceiptVerifierParameters::default();
    let suc = SuccinctReceiptVerifierParameters::default();
    println!("risc0-zkvm {}", risc0_zkvm::VERSION);
    println!(
        "risc0 packages in Cargo.lock (all crates.io, checked by build.rs): {}",
        option_env!("STOCK_VERIFY31_RISC0_PACKAGES").unwrap_or("unchecked")
    );
    println!("ALLOWED_CONTROL_ROOT {}", short(&ALLOWED_CONTROL_ROOT));
    println!(
        "BN254_IDENTITY_CONTROL_ID {}",
        short(&risc0_circuit_recursion::control_id::BN254_IDENTITY_CONTROL_ID)
    );
    println!(
        "Groth16ReceiptVerifierParameters::default() digest {} (selector {}), control_root {}, \
         bn254_control_id {}",
        short(&g16.digest()),
        hex::encode(&g16.digest().as_bytes()[..4]),
        short(&g16.control_root),
        short(&g16.bn254_control_id)
    );
    println!(
        "SuccinctReceiptVerifierParameters::default() digest {}, control_root {}, proof system {}, \
         circuit {}",
        short(&suc.digest()),
        short(&suc.control_root),
        suc.proof_system_info,
        suc.circuit_info
    );
    println!(
        "SegmentReceiptVerifierParameters::default() digest {}, {} control ids, proof system {}, \
         circuit {}",
        short(&seg.digest()),
        seg.control_ids.len(),
        seg.proof_system_info,
        seg.circuit_info
    );
    println!(
        "CompositeReceiptVerifierParameters::default() digest {}",
        short(&CompositeReceiptVerifierParameters::default().digest())
    );
    println!(
        "RV32IM_SEAL_VERSION {}",
        risc0_circuit_rv32im::RV32IM_SEAL_VERSION
    );
}
