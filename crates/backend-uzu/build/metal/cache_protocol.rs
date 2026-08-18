use std::{
    collections::HashMap,
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};

#[cfg(test)]
use super::caching;
#[cfg(not(test))]
use crate::common::caching;

pub const ZSTD_CODEC_ID: &str = "zstd-encode_all";
pub const STAGE_LOCK_TIMEOUT: Duration = Duration::from_secs(20 * 60);
const LOCK_OWNER_GRACE: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    Air,
    Metallib,
    Zstd,
    Index,
}

impl Stage {
    pub fn dir_name(self) -> &'static str {
        match self {
            Self::Air => "air",
            Self::Metallib => "metallib",
            Self::Zstd => "zstd",
            Self::Index => "index",
        }
    }
}

#[derive(Serialize, Deserialize)]
struct StageManifest {
    schema: u32,
    kind: String,
    hash: [u8; blake3::OUT_LEN],
    len: u64,
}

#[derive(Clone, Debug)]
pub struct SharedArtifactCache {
    root: PathBuf,
}

pub struct StageLock {
    path: PathBuf,
}

/// Immutable stage output after manifest and content verification.
pub struct CachedArtifact {
    pub path: PathBuf,
    pub hash: [u8; blake3::OUT_LEN],
}

impl Drop for StageLock {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

impl SharedArtifactCache {
    pub fn new() -> anyhow::Result<Self> {
        Self::at(caching::cargo_target_dir()?.join("uzu-metal-cache").join(format!("v{}", caching::METAL_CACHE_SCHEMA)))
    }

    pub fn at(root: impl Into<PathBuf>) -> anyhow::Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root).with_context(|| format!("cannot create {}", root.display()))?;
        Ok(Self {
            root,
        })
    }

    pub fn stage_dir(
        &self,
        stage: Stage,
        key: &blake3::Hash,
    ) -> PathBuf {
        self.root.join(stage.dir_name()).join(key.to_hex().as_str())
    }

    pub async fn lock(
        &self,
        stage: Stage,
        key: &blake3::Hash,
    ) -> anyhow::Result<StageLock> {
        self.lock_with_timeout(stage, key, STAGE_LOCK_TIMEOUT).await
    }

    pub async fn lock_with_timeout(
        &self,
        stage: Stage,
        key: &blake3::Hash,
        timeout: Duration,
    ) -> anyhow::Result<StageLock> {
        let dir = self.stage_dir(stage, key);
        fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
        let lock_path = dir.join(".lock");
        let deadline = Instant::now() + timeout;
        let mut sleep = Duration::from_millis(20);
        loop {
            match fs::create_dir(&lock_path) {
                Ok(()) => {
                    let owner_path = lock_path.join("pid");
                    let owner = fs::write(&owner_path, std::process::id().to_string());
                    if let Err(err) = owner {
                        let _ = fs::remove_dir_all(&lock_path);
                        return Err(err).with_context(|| format!("cannot write {}", owner_path.display()));
                    }
                    return Ok(StageLock {
                        path: lock_path,
                    });
                },
                Err(err) if err.kind() == ErrorKind::AlreadyExists => {
                    if lock_holder_dead(&lock_path) {
                        let _ = fs::remove_dir_all(&lock_path);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        anyhow::bail!("timed out waiting for metal cache lock {}", lock_path.display());
                    }
                    tokio::time::sleep(sleep).await;
                    sleep = (sleep * 2).min(Duration::from_millis(250));
                },
                Err(err) => {
                    return Err(err).with_context(|| format!("cannot lock {}", lock_path.display()));
                },
            }
        }
    }

    pub fn lookup(
        &self,
        stage: Stage,
        key: &blake3::Hash,
    ) -> anyhow::Result<Option<CachedArtifact>> {
        let dir = self.stage_dir(stage, key);
        let manifest_path = dir.join("manifest.json");
        let artifact = dir.join("artifact");
        let Ok(bytes) = fs::read(&manifest_path) else {
            return Ok(None);
        };
        let Ok(manifest) = serde_json::from_slice::<StageManifest>(&bytes) else {
            return Ok(None);
        };
        if manifest.schema != caching::METAL_CACHE_SCHEMA || manifest.kind != stage.dir_name() {
            return Ok(None);
        }
        let Ok(meta) = fs::metadata(&artifact) else {
            return Ok(None);
        };
        if meta.len() != manifest.len {
            return Ok(None);
        }
        if caching::hash_file(&artifact)?.as_bytes() != &manifest.hash {
            return Ok(None);
        }
        Ok(Some(CachedArtifact {
            path: artifact,
            hash: manifest.hash,
        }))
    }

    pub fn store(
        &self,
        stage: Stage,
        key: &blake3::Hash,
        src: impl AsRef<Path>,
    ) -> anyhow::Result<[u8; blake3::OUT_LEN]> {
        let dir = self.stage_dir(stage, key);
        fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
        let artifact = dir.join("artifact");
        caching::copy_atomic(src.as_ref(), &artifact)?;
        self.commit_manifest(stage, &artifact)
    }

    pub fn store_bytes(
        &self,
        stage: Stage,
        key: &blake3::Hash,
        bytes: impl AsRef<[u8]>,
    ) -> anyhow::Result<[u8; blake3::OUT_LEN]> {
        let dir = self.stage_dir(stage, key);
        fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
        let artifact = dir.join("artifact");
        caching::write_atomic(&artifact, bytes)?;
        self.commit_manifest(stage, &artifact)
    }

    fn commit_manifest(
        &self,
        stage: Stage,
        artifact: &Path,
    ) -> anyhow::Result<[u8; blake3::OUT_LEN]> {
        let hash = caching::hash_file(artifact)?;
        let len = fs::metadata(artifact).with_context(|| format!("cannot stat {}", artifact.display()))?.len();
        let manifest = StageManifest {
            schema: caching::METAL_CACHE_SCHEMA,
            kind: stage.dir_name().to_string(),
            hash: *hash.as_bytes(),
            len,
        };
        caching::write_atomic(
            artifact.with_file_name("manifest.json"),
            serde_json::to_vec_pretty(&manifest).context("cannot serialize stage manifest")?,
        )?;
        Ok(*hash.as_bytes())
    }
}

pub fn air_key(
    source_path_relative: &str,
    dependency_hashes: &HashMap<Box<str>, [u8; blake3::OUT_LEN]>,
    footer: &str,
    compile_schema_hash: &blake3::Hash,
    toolchain_hash: &blake3::Hash,
) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"uzu-metal-air-v2");
    hasher.update(&caching::METAL_CACHE_SCHEMA.to_le_bytes());
    hasher.update(source_path_relative.as_bytes());
    hasher.update(b"\0");
    update_sorted_deps(&mut hasher, dependency_hashes);
    hasher.update(footer.as_bytes());
    hasher.update(compile_schema_hash.as_bytes());
    hasher.update(toolchain_hash.as_bytes());
    hasher.finalize()
}

pub fn metallib_key(
    air_hash: &blake3::Hash,
    linker_hash: &blake3::Hash,
) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"uzu-metal-metallib-v2");
    hasher.update(&caching::METAL_CACHE_SCHEMA.to_le_bytes());
    hasher.update(air_hash.as_bytes());
    hasher.update(linker_hash.as_bytes());
    hasher.finalize()
}

pub fn zstd_key(
    metallib_hash: &blake3::Hash,
    zstd_level: i32,
) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"uzu-metal-zstd-v2");
    hasher.update(&caching::METAL_CACHE_SCHEMA.to_le_bytes());
    hasher.update(metallib_hash.as_bytes());
    hasher.update(ZSTD_CODEC_ID.as_bytes());
    hasher.update(&zstd_level.to_le_bytes());
    hasher.finalize()
}

pub fn index_key(
    source_path_relative: &str,
    source_hash: &blake3::Hash,
    compile_schema_hash: &blake3::Hash,
    analyzer_hash: &blake3::Hash,
    gpu_types_hash: &blake3::Hash,
) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"uzu-metal-index-v2");
    hasher.update(&caching::METAL_CACHE_SCHEMA.to_le_bytes());
    hasher.update(source_path_relative.as_bytes());
    hasher.update(b"\0");
    hasher.update(source_hash.as_bytes());
    hasher.update(compile_schema_hash.as_bytes());
    hasher.update(analyzer_hash.as_bytes());
    hasher.update(gpu_types_hash.as_bytes());
    hasher.finalize()
}

pub fn dependencies_match(stored: &HashMap<Box<str>, [u8; blake3::OUT_LEN]>) -> bool {
    stored.iter().all(|(path, hash)| fs::read(path.as_ref()).is_ok_and(|bytes| blake3::hash(&bytes).as_bytes() == hash))
}

fn update_sorted_deps(
    hasher: &mut blake3::Hasher,
    dependency_hashes: &HashMap<Box<str>, [u8; blake3::OUT_LEN]>,
) {
    let mut deps: Vec<_> = dependency_hashes.iter().collect();
    deps.sort_by(|a, b| a.0.cmp(b.0));
    for (path, hash) in deps {
        hasher.update(path.as_bytes());
        hasher.update(b"\0");
        hasher.update(hash);
    }
}

fn lock_holder_dead(lock_path: &Path) -> bool {
    let Ok(pid) = fs::read_to_string(lock_path.join("pid")) else {
        let age = fs::metadata(lock_path)
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| std::time::SystemTime::now().duration_since(modified).ok());
        return age.is_some_and(|age| age >= LOCK_OWNER_GRACE);
    };
    let Ok(pid) = pid.trim().parse::<u32>() else {
        return true;
    };
    process_exists(pid).is_ok_and(|exists| !exists)
}

fn process_exists(pid: u32) -> std::io::Result<bool> {
    let status = std::process::Command::new("/bin/kill")
        .args(["-0", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    Ok(status.success())
}
