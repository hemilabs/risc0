// Refuse to build unless every risc0 package in Cargo.lock comes from the crates.io registry, the
// manifest has no path/git dependency, [patch] or [replace], and no cargo config file that cargo
// reads for this build (the ancestors of this directory, and CARGO_HOME) has a `paths` override
// or a [patch] (a `paths` override would swap in local sources while Cargo.lock still shows the
// registry). This is what makes the binary a judge that is independent of the risc0 tree it lives
// in. The committed Cargo.lock is required: without it there is nothing to check.

use std::{
    fs,
    path::{Path, PathBuf},
};

const REGISTRY: &str = "registry+https://github.com/rust-lang/crates.io-index";

fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let dir = Path::new(&dir);
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=Cargo.toml");

    let manifest = fs::read_to_string(dir.join("Cargo.toml")).unwrap();
    for line in manifest.lines() {
        let tight: String = line.split('#').next().unwrap().split_whitespace().collect();
        assert!(
            !tight.starts_with("[patch")
                && !tight.starts_with("[replace")
                && !tight.contains("path=")
                && !tight.contains("git="),
            "stock_verify31/Cargo.toml must not patch, replace, path- or git-depend: {line}"
        );
    }

    for config in cargo_configs(dir) {
        println!("cargo:rerun-if-changed={}", config.display());
        let Ok(text) = fs::read_to_string(&config) else {
            continue;
        };
        for line in text.lines() {
            let tight: String = line.split('#').next().unwrap().split_whitespace().collect();
            assert!(
                !tight.starts_with("paths=") && !tight.starts_with("[patch"),
                "{} overrides package sources ({line}): the stock verifier would not be stock",
                config.display()
            );
        }
    }

    let lock = fs::read_to_string(dir.join("Cargo.lock")).unwrap_or_else(|e| {
        panic!("stock_verify31/Cargo.lock is required (it is committed; restore it from git): {e}")
    });
    let mut risc0 = 0;
    for pkg in lock.split("[[package]]").skip(1) {
        let field = |k: &str| {
            pkg.lines()
                .find_map(|l| l.strip_prefix(&format!("{k} = \"")))
                .map(|v| v.trim_end_matches('"').to_string())
        };
        let name = field("name").unwrap_or_default();
        let source = field("source");
        if name == "stock-verify31" {
            assert!(source.is_none());
            continue;
        }
        if name.starts_with("risc0") || name.starts_with("bonsai") {
            risc0 += 1;
            assert_eq!(
                source.as_deref(),
                Some(REGISTRY),
                "{name} must come from crates.io, Cargo.lock says {source:?}"
            );
        }
        assert!(
            source.is_some(),
            "{name} has no source in Cargo.lock (a path dependency)"
        );
    }
    assert!(risc0 > 0, "no risc0 package in Cargo.lock");
    let zkvm = lock
        .split("[[package]]")
        .find(|p| p.contains("name = \"risc0-zkvm\"\n"))
        .expect("risc0-zkvm in Cargo.lock");
    assert!(
        zkvm.contains("version = \"3.1.0\"\n"),
        "risc0-zkvm must be 3.1.0"
    );
    println!("cargo:rustc-env=STOCK_VERIFY31_RISC0_PACKAGES={risc0}");
}

/// The cargo config files cargo may read for a build in `dir`: `.cargo/config{,.toml}` in `dir`
/// and every ancestor, and `config{,.toml}` in CARGO_HOME (default `~/.cargo`).
fn cargo_configs(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for d in dir.ancestors() {
        out.push(d.join(".cargo/config"));
        out.push(d.join(".cargo/config.toml"));
    }
    let home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cargo")));
    if let Some(home) = home {
        out.push(home.join("config"));
        out.push(home.join("config.toml"));
    }
    out
}
