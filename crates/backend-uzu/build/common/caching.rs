#![cfg(all(feature = "metal", target_os = "macos"))]

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::Context;
use walkdir::WalkDir;

pub const METAL_CACHE_SCHEMA: u32 = 2;

fn sibling_tmp(path: &Path) -> PathBuf {
    let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_else(|| "file".to_string());
    path.with_file_name(format!(".{name}.tmp-{}", std::process::id()))
}

pub fn write_atomic(
    path: impl AsRef<Path>,
    contents: impl AsRef<[u8]>,
) -> anyhow::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let tmp = sibling_tmp(path);
    {
        let mut file = fs::File::create(&tmp).with_context(|| format!("cannot create {}", tmp.display()))?;
        file.write_all(contents.as_ref()).with_context(|| format!("cannot write {}", tmp.display()))?;
        file.sync_all().with_context(|| format!("cannot sync {}", tmp.display()))?;
    }
    fs::rename(&tmp, path).with_context(|| format!("cannot rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

pub fn copy_atomic(
    src: impl AsRef<Path>,
    dest: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let dest = dest.as_ref();
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let tmp = sibling_tmp(dest);
    fs::copy(src.as_ref(), &tmp).with_context(|| format!("cannot copy to {}", tmp.display()))?;
    fs::rename(&tmp, dest).with_context(|| format!("cannot rename {} -> {}", tmp.display(), dest.display()))?;
    Ok(())
}

/// Target-local cache materialization without copying large Metal artifacts.
pub fn hard_link_atomic(
    src: impl AsRef<Path>,
    dest: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let dest = dest.as_ref();
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let tmp = sibling_tmp(dest);
    if let Err(err) = fs::remove_file(&tmp)
        && err.kind() != std::io::ErrorKind::NotFound
    {
        return Err(err).with_context(|| format!("cannot remove stale {}", tmp.display()));
    }
    fs::hard_link(src.as_ref(), &tmp).with_context(|| format!("cannot link to {}", tmp.display()))?;
    fs::rename(&tmp, dest).with_context(|| format!("cannot rename {} -> {}", tmp.display(), dest.display()))?;
    Ok(())
}

/// Removes an output before a compiler can truncate a cache-backed hard link.
pub fn remove_file_if_exists(path: impl AsRef<Path>) -> anyhow::Result<()> {
    let path = path.as_ref();
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err).with_context(|| format!("cannot remove {}", path.display())),
    }
}

pub fn hash_file(path: impl AsRef<Path>) -> anyhow::Result<blake3::Hash> {
    let path = path.as_ref();
    Ok(blake3::hash(&fs::read(path).with_context(|| format!("cannot read {}", path.display()))?))
}

pub fn hash_paths(paths: impl IntoIterator<Item = impl AsRef<Path>>) -> anyhow::Result<blake3::Hash> {
    let mut files = Vec::new();
    for path in paths {
        let path = path.as_ref();
        if path.is_dir() {
            for entry in WalkDir::new(path) {
                let entry = entry.with_context(|| format!("cannot walk {}", path.display()))?;
                if entry.file_type().is_file() {
                    files.push(entry.into_path());
                }
            }
        } else if path.is_file() {
            files.push(path.to_path_buf());
        }
    }
    files.sort();
    let mut hasher = blake3::Hasher::new();
    for file in &files {
        hasher.update(file.to_string_lossy().as_bytes());
        hasher.update(b"\0");
        hasher.update(&fs::read(file).with_context(|| format!("cannot read {}", file.display()))?);
    }
    Ok(hasher.finalize())
}

pub fn cargo_target_dir() -> anyhow::Result<PathBuf> {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("missing OUT_DIR")?);
    let out_dir = fs::canonicalize(&out_dir).with_context(|| format!("cannot resolve {}", out_dir.display()))?;
    for ancestor in out_dir.ancestors() {
        if ancestor.file_name().is_some_and(|name| name == "build")
            && let Some(target_dir) = ancestor.parent().and_then(Path::parent)
        {
            return Ok(target_dir.to_path_buf());
        }
    }
    anyhow::bail!("cannot derive Cargo target dir from {}", out_dir.display())
}
