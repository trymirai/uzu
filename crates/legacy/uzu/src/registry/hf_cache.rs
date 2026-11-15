use std::{
    collections::HashSet,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
};

use hf_hub::{HFClient, HFError};
use kiban::{fs::asyn as fs, rt::MakeSendFuture};
use shoji::{
    traits::Registry as RegistryTrait,
    types::model::{Model, ModelAccessibility, ModelSource, ModelSpecialization},
};
use uzu_engine::engine::{ModelType, resolve_model_type};

use crate::registry::{RegistryError, local::load_encoding};

pub struct Registry {
    cache_path: PathBuf,
    backend_identifier: String,
    backend_version: String,
}

impl Registry {
    pub fn new(
        backend_identifier: String,
        backend_version: String,
    ) -> Self {
        let cache_path = HFClient::new()
            .map(|client| client.cache_dir().to_path_buf())
            .unwrap_or_else(|_| default_hf_cache_path());
        Self::with_cache_path(cache_path, backend_identifier, backend_version)
    }

    pub fn with_cache_path(
        cache_path: PathBuf,
        backend_identifier: String,
        backend_version: String,
    ) -> Self {
        Self {
            cache_path,
            backend_identifier,
            backend_version,
        }
    }

    pub fn cache_path(&self) -> &Path {
        &self.cache_path
    }
}

impl RegistryTrait for Registry {
    type Error = RegistryError;

    fn identifier(&self) -> String {
        format!("hf-cache:{}", self.cache_path.display())
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
        Box::pin(MakeSendFuture::new(async {
            if !fs::is_dir(&self.cache_path).await {
                return Ok(Vec::new());
            }

            let mut discovered: Vec<(String, PathBuf)> = Vec::new();
            let mut seen_repos: HashSet<String> = HashSet::new();

            // 1. Primary scan via official hf-hub cache scanner.
            if let Ok(client) = HFClient::builder().cache_dir(&self.cache_path).build()
                && let Ok(cache_info) = client.scan_cache().send().await {
                    for repo in cache_info.repos {
                        if repo.repo_type != "model" {
                            continue;
                        }
                        if let Some(snapshot_dir) = pick_cached_revision(&repo).await {
                            seen_repos.insert(repo.repo_id.clone());
                            discovered.push((repo.repo_id, snapshot_dir));
                        }
                    }
                }

            // 2. Fallback scan for manually linked local snapshots (e.g. `models--local--<name>/snapshots/local -> /path/to/model`)
            //    which may not use `blobs/` and thus can be skipped by `scan_cache()`.
            if let Ok(entries) = fs::read_dir(&self.cache_path).await {
                for entry_path in entries {
                    if !fs::is_dir(&entry_path).await {
                        continue;
                    }
                    let Some(dir_name) = entry_path.file_name().and_then(|name| name.to_str()) else {
                        continue;
                    };
                    let Some(repo_id) = parse_hf_repo_dir_name(dir_name) else {
                        continue;
                    };
                    if seen_repos.contains(&repo_id) {
                        continue;
                    }
                    if let Some(snapshot_dir) = resolve_snapshot_dir(&entry_path).await {
                        seen_repos.insert(repo_id.clone());
                        discovered.push((repo_id, snapshot_dir));
                    }
                }
            }

            let mut models = Vec::new();
            for (repo_id, snapshot_dir) in discovered {
                match self.model(&repo_id, &snapshot_dir).await {
                    Ok(model) => models.push(model),
                    Err(error) => {
                        tracing::warn!(?error, path = %snapshot_dir.display(), "skipping invalid HF cached model");
                    },
                }
            }

            Ok(models)
        }))
    }
}

impl Registry {
    async fn model(
        &self,
        repo_id: &str,
        snapshot_dir: &Path,
    ) -> Result<Model, RegistryError> {
        let specialization = resolve_model_type(snapshot_dir)
            .await
            .map(|model_type| match model_type {
                ModelType::LanguageModel => ModelSpecialization::Chat {},
                ModelType::Classifier => ModelSpecialization::Classification {},
                ModelType::Decision => ModelSpecialization::Decision {},
            })
            .map_err(|error| RegistryError::UnableToGetModels {
                message: format!("Unable to resolve specialization for {}: {error}", snapshot_dir.display()),
            })?;

        Ok(Model::external(
            repo_id.to_string(),
            self.identifier(),
            "Hugging Face Cache".to_string(),
            self.backend_identifier.clone(),
            self.backend_identifier.clone(),
            self.backend_version.clone(),
            vec![specialization],
            ModelAccessibility::OnDevice {
                source: ModelSource::Filesystem {
                    path: snapshot_dir.to_string_lossy().to_string(),
                },
            },
            load_encoding(snapshot_dir).await,
        ))
    }
}

fn default_hf_cache_path() -> PathBuf {
    if let Ok(hub_cache) = std::env::var("HF_HUB_CACHE") {
        let p = PathBuf::from(hub_cache);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        let p = PathBuf::from(hf_home);
        if !p.as_os_str().is_empty() {
            return p.join("hub");
        }
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache").join("huggingface").join("hub")
}

async fn pick_cached_revision(repo: &hf_hub::cache::CachedRepoInfo) -> Option<PathBuf> {
    if let Some(main_rev) = repo.revisions.iter().find(|rev| rev.refs.iter().any(|r| r == "main"))
        && fs::is_file(main_rev.snapshot_path.join("config.json")).await {
            return Some(main_rev.snapshot_path.clone());
        }
    let mut revs: Vec<&hf_hub::cache::CachedRevisionInfo> = repo.revisions.iter().collect();
    revs.sort_by_key(|rev| std::cmp::Reverse(rev.last_modified));
    for rev in revs {
        if fs::is_file(rev.snapshot_path.join("config.json")).await {
            return Some(rev.snapshot_path.clone());
        }
    }
    None
}

/// Parse `models--<org>--<repo>` into `<org>/<repo>`.
pub fn parse_hf_repo_dir_name(dir_name: &str) -> Option<String> {
    let rest = dir_name.strip_prefix("models--")?;
    let (org, repo) = rest.split_once("--")?;
    if org.is_empty() || repo.is_empty() {
        return None;
    }
    Some(format!("{org}/{repo}"))
}

async fn resolve_snapshot_dir(repo_cache_dir: &Path) -> Option<PathBuf> {
    let snapshots_dir = repo_cache_dir.join("snapshots");
    if !fs::is_dir(&snapshots_dir).await {
        return None;
    }

    if let Ok(rev) = fs::read_to_string(repo_cache_dir.join("refs").join("main")).await {
        let rev = rev.trim();
        if !rev.is_empty() {
            let candidate = snapshots_dir.join(rev);
            if fs::is_dir(&candidate).await && fs::is_file(candidate.join("config.json")).await {
                return Some(candidate);
            }
        }
    }

    let entries = fs::read_dir(&snapshots_dir).await.ok()?;
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for candidate in entries {
        if !fs::is_dir(&candidate).await || !fs::is_file(candidate.join("config.json")).await {
            continue;
        }
        let modified = fs::file_modified(&candidate).await.unwrap_or(std::time::UNIX_EPOCH);
        if best.as_ref().is_none_or(|(best_time, _)| modified > *best_time) {
            best = Some((modified, candidate));
        }
    }
    best.map(|(_, path)| path)
}

/// Returns true if `value` looks like `<org>/<repo>` (a Hugging Face repository ID)
/// rather than a filesystem path or catalog identifier.
pub fn looks_like_hf_repo_id(value: &str) -> bool {
    if value.starts_with('/') || value.starts_with('.') || value.starts_with('~') {
        return false;
    }
    if value.contains(':') || value.chars().any(char::is_whitespace) {
        return false;
    }
    let Some((org, repo)) = value.split_once('/') else {
        return false;
    };
    !org.is_empty() && !repo.is_empty() && !repo.contains('/')
}

fn build_hf_client(token: Option<&str>) -> Result<HFClient, RegistryError> {
    let mut builder = HFClient::builder();
    if let Some(token) = token {
        builder = builder.token(token);
    }
    builder.build().map_err(|error| RegistryError::UnableToCreate {
        message: error.to_string(),
    })
}

/// Check whether `repo_id` (`<org>/<repo>`) exists and is accessible on Hugging Face Hub.
pub async fn hf_repo_exists(
    repo_id: &str,
    token: Option<&str>,
) -> Result<bool, RegistryError> {
    let Some((org, repo)) = repo_id.split_once('/') else {
        return Ok(false);
    };
    let client = build_hf_client(token)?;
    match client.model(org, repo).info().send().await {
        Ok(_) => Ok(true),
        Err(HFError::RepoNotFound { .. } | HFError::AuthRequired { .. } | HFError::Forbidden { .. }) => Ok(false),
        Err(error) => Err(RegistryError::UnableToGetModels {
            message: format!("Failed to query Hugging Face for `{repo_id}`: {error}"),
        }),
    }
}

/// Download a snapshot of `repo_id` (`<org>/<repo>`) into the standard Hugging Face Hub cache
/// and return the resolved snapshot directory path.
pub async fn download_hf_repo(
    repo_id: &str,
    token: Option<&str>,
) -> Result<PathBuf, RegistryError> {
    let Some((org, repo)) = repo_id.split_once('/') else {
        return Err(RegistryError::UnableToGetModels {
            message: format!("Invalid Hugging Face repo id: `{repo_id}`"),
        });
    };
    let client = build_hf_client(token)?;
    client
        .model(org, repo)
        .snapshot_download()
        .send()
        .await
        .map_err(|error| RegistryError::UnableToGetModels {
            message: format!("Failed to download `{repo_id}` from Hugging Face: {error}"),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hf_repo_dir_name() {
        assert_eq!(
            parse_hf_repo_dir_name("models--davidxifeng--LiquidAI-LFM3-Audio-2B-MLX-4bit-uzu").as_deref(),
            Some("davidxifeng/LiquidAI-LFM3-Audio-2B-MLX-4bit-uzu")
        );
        assert_eq!(parse_hf_repo_dir_name("models--local--Haru-4B-8bit-uzu").as_deref(), Some("local/Haru-4B-8bit-uzu"));
        assert_eq!(parse_hf_repo_dir_name("datasets--foo--bar"), None);
        assert_eq!(parse_hf_repo_dir_name("models--invalid"), None);
    }

    #[test]
    fn detects_hf_repo_id_shape() {
        assert!(looks_like_hf_repo_id("davidxifeng/LiquidAI-LFM3-Audio-2B-MLX-4bit-uzu"));
        assert!(!looks_like_hf_repo_id("/path/to/models/lfm3"));
        assert!(!looks_like_hf_repo_id("./models/lfm3"));
        assert!(!looks_like_hf_repo_id("~/models/lfm3"));
        assert!(!looks_like_hf_repo_id("alibaba:qwen3.5:0.8b"));
        assert!(!looks_like_hf_repo_id("single-name"));
    }

    #[tokio::test]
    async fn discovers_symlinked_and_refs_main_snapshots() {
        let root = tempfile::tempdir().unwrap();
        let repo_dir = root.path().join("models--local--TestModel");
        let refs_dir = repo_dir.join("refs");
        let snapshots_dir = repo_dir.join("snapshots");
        std::fs::create_dir_all(&refs_dir).unwrap();
        std::fs::create_dir_all(&snapshots_dir).unwrap();
        std::fs::write(refs_dir.join("main"), "local\n").unwrap();

        let target_dir = tempfile::tempdir().unwrap();
        // Minimal valid LanguageModelConfig JSON is not needed for snapshot resolution test;
        // verify resolve_snapshot_dir finds config.json through symlinked snapshot dir.
        std::fs::write(target_dir.path().join("config.json"), b"{}").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(target_dir.path(), snapshots_dir.join("local")).unwrap();
        #[cfg(not(unix))]
        std::fs::create_dir_all(snapshots_dir.join("local")).unwrap();
        #[cfg(not(unix))]
        std::fs::write(snapshots_dir.join("local").join("config.json"), b"{}").unwrap();

        let resolved = resolve_snapshot_dir(&repo_dir).await;
        assert_eq!(resolved.as_deref(), Some(snapshots_dir.join("local").as_path()));
    }
}
