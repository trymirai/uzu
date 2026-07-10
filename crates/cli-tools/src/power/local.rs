use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use backend_uzu::{HeaderSummary, summarize_header};

use super::artifacts::{CONFIG_FILE, cache_models_path, resolve_weights_path};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalArtifact {
    pub id: String,
    pub model_dir: PathBuf,
    pub config_path: PathBuf,
    pub header_path: PathBuf,
    pub header_summary: HeaderSummary,
}

pub fn discover(
    storage_base: &Path,
    model_ids: &[String],
) -> Result<Vec<LocalArtifact>> {
    let models_root = cache_models_path(storage_base);
    if !models_root.is_dir() {
        bail!("models directory missing: {}", models_root.display());
    }

    let mut artifacts = Vec::new();
    collect_artifacts(&models_root, &models_root, model_ids, &mut artifacts)?;
    artifacts.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(artifacts)
}

fn collect_artifacts(
    models_root: &Path,
    current: &Path,
    model_ids: &[String],
    artifacts: &mut Vec<LocalArtifact>,
) -> Result<()> {
    let config_path = current.join(CONFIG_FILE);
    if config_path.is_file() {
        let Some(header_path) = resolve_weights_path(current) else {
            bail!(
                "config.json at {} is missing {} or {}",
                current.display(),
                super::artifacts::HEADER_FILE,
                super::artifacts::WEIGHTS_FILE
            );
        };
        let id = current
            .strip_prefix(models_root)
            .with_context(|| format!("strip prefix from {}", current.display()))?
            .to_string_lossy()
            .replace('\\', "/");
        if !matches_model_ids(model_ids, &id) {
            return Ok(());
        }
        let header_summary =
            summarize_header(&header_path).with_context(|| format!("summarize header {}", header_path.display()))?;
        artifacts.push(LocalArtifact {
            id,
            model_dir: current.to_path_buf(),
            config_path,
            header_path,
            header_summary,
        });
        return Ok(());
    }

    let entries = std::fs::read_dir(current).with_context(|| format!("read {}", current.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("read entry under {}", current.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_artifacts(models_root, &path, model_ids, artifacts)?;
        }
    }
    Ok(())
}

fn matches_model_ids(
    selected: &[String],
    id: &str,
) -> bool {
    selected.is_empty() || selected.iter().any(|candidate| candidate == id)
}
