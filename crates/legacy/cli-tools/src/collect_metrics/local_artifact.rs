use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use uzu_engine::HeaderSummary;

use super::{
    artifacts::{CONFIG_FILE, HEADER_FILE, TOKENIZER_FILE, WEIGHTS_FILE, cache_models_path, resolve_weights_path},
    options::Options,
};

/// A model directory discovered under a storage tree, replayable without network access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalArtifact {
    pub id: String,
    pub model_dir: PathBuf,
    pub header_path: PathBuf,
    pub header_summary: HeaderSummary,
}

impl LocalArtifact {
    pub fn discover(
        storage_base: &Path,
        options: &Options,
    ) -> Result<Vec<Self>> {
        let models_root = cache_models_path(storage_base);
        if !models_root.is_dir() {
            bail!("models directory missing: {}", models_root.display());
        }

        let mut artifacts = Vec::new();
        Self::collect(&models_root, &models_root, options, &mut artifacts)?;
        artifacts.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(artifacts)
    }

    fn collect(
        models_root: &Path,
        current: &Path,
        options: &Options,
        artifacts: &mut Vec<Self>,
    ) -> Result<()> {
        if current.join(CONFIG_FILE).is_file() {
            let id = current
                .strip_prefix(models_root)
                .with_context(|| format!("strip prefix from {}", current.display()))?
                .to_string_lossy()
                .replace('\\', "/");
            if !options.selects(&id) {
                return Ok(());
            }
            if !current.join(TOKENIZER_FILE).is_file() {
                bail!("{} at {} is missing {}", CONFIG_FILE, current.display(), TOKENIZER_FILE);
            }
            let Some(header_path) = resolve_weights_path(current) else {
                bail!("{} at {} is missing {} or {}", CONFIG_FILE, current.display(), HEADER_FILE, WEIGHTS_FILE);
            };
            let header_summary = HeaderSummary::read(&header_path)
                .with_context(|| format!("summarize header {}", header_path.display()))?;
            artifacts.push(Self {
                id,
                model_dir: current.to_path_buf(),
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
                Self::collect(models_root, &path, options, artifacts)?;
            }
        }
        Ok(())
    }
}
