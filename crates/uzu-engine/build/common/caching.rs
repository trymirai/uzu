#![cfg(all(feature = "metal", target_os = "macos"))]

use std::{
    fs,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::Context;
use walkdir::WalkDir;

static BUILD_SYSTEM_HASH: OnceLock<blake3::Hash> = OnceLock::new();

fn hash_directory(
    hasher: &mut blake3::Hasher,
    directory: &Path,
) -> anyhow::Result<()> {
    let mut files: Vec<PathBuf> = WalkDir::new(directory)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("cannot walk {}", directory.display()))?
        .into_iter()
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .collect();
    files.sort();

    for file in files {
        hash_file(hasher, &file)?;
    }

    Ok(())
}

fn hash_file(
    hasher: &mut blake3::Hasher,
    file: &Path,
) -> anyhow::Result<()> {
    hasher.update(file.as_os_str().as_encoded_bytes());
    hasher.update(b"\0");
    hasher.update(&fs::read(file).with_context(|| format!("cannot read {}", file.display()))?);
    hasher.update(b"\0");

    Ok(())
}

pub fn build_system_hash() -> anyhow::Result<&'static blake3::Hash> {
    if let Some(bsh) = BUILD_SYSTEM_HASH.get() {
        Ok(bsh)
    } else {
        let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?);
        let workspace_dir =
            crate_dir.parent().and_then(Path::parent).context("cannot find workspace root from CARGO_MANIFEST_DIR")?;

        let mut hasher = blake3::Hasher::new();
        hash_file(&mut hasher, &workspace_dir.join("Cargo.toml")).context("cannot hash Cargo.toml")?;
        hash_file(&mut hasher, &workspace_dir.join("Cargo.lock")).context("cannot hash Cargo.lock")?;
        hash_file(&mut hasher, &crate_dir.join("Cargo.toml")).context("cannot hash Cargo.toml")?;
        hash_directory(&mut hasher, &crate_dir.join("build")).context("cannot hash build/")?;

        Ok(BUILD_SYSTEM_HASH.get_or_init(|| hasher.finalize()))
    }
}
