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

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::power::artifacts::{HEADER_FILE, WEIGHTS_FILE};

    fn write_min_header(path: &Path) {
        let json = br#"{"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut bytes = Vec::with_capacity(8 + json.len());
        bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(json);
        fs::write(path, bytes).expect("write header");
    }

    fn write_model_tree(
        storage_base: &Path,
        relative_id: &str,
        header_name: &str,
    ) -> PathBuf {
        let model_dir = cache_models_path(storage_base).join(relative_id);
        fs::create_dir_all(&model_dir).expect("create model dir");
        fs::write(model_dir.join(CONFIG_FILE), "{}").expect("write config");
        write_min_header(&model_dir.join(header_name));
        model_dir
    }

    #[test]
    fn discovers_header_only_artifacts_with_stable_ids() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_model_tree(temp.path(), "alpha/model-a/v1", HEADER_FILE);
        write_model_tree(temp.path(), "beta/model-b/v2", WEIGHTS_FILE);

        let artifacts = discover(temp.path(), &[]).expect("discover");
        assert_eq!(artifacts.len(), 2);
        assert_eq!(artifacts[0].id, "alpha/model-a/v1");
        assert_eq!(artifacts[1].id, "beta/model-b/v2");
        assert_eq!(artifacts[0].header_summary.tensor_count, 1);
        assert_eq!(artifacts[0].header_summary.logical_payload_bytes, 16);
    }

    #[test]
    fn filters_by_exact_storage_relative_id() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_model_tree(temp.path(), "alpha/model-a/v1", HEADER_FILE);
        write_model_tree(temp.path(), "beta/model-b/v2", HEADER_FILE);

        let artifacts = discover(temp.path(), &["alpha/model-a/v1".to_string()]).expect("discover");
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].id, "alpha/model-a/v1");
    }

    #[test]
    fn accepts_multiple_model_ids() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_model_tree(temp.path(), "alpha/model-a/v1", HEADER_FILE);
        write_model_tree(temp.path(), "beta/model-b/v2", HEADER_FILE);
        write_model_tree(temp.path(), "gamma/model-c/v3", HEADER_FILE);

        let artifacts =
            discover(temp.path(), &["alpha/model-a/v1".to_string(), "gamma/model-c/v3".to_string()]).expect("discover");
        assert_eq!(artifacts.len(), 2);
        assert_eq!(artifacts[0].id, "alpha/model-a/v1");
        assert_eq!(artifacts[1].id, "gamma/model-c/v3");
    }

    #[test]
    fn missing_weights_file_is_an_error() {
        let temp = tempfile::tempdir().expect("tempdir");
        let model_dir = cache_models_path(temp.path()).join("alpha/model-a/v1");
        fs::create_dir_all(&model_dir).expect("create model dir");
        fs::write(model_dir.join(CONFIG_FILE), "{}").expect("write config");

        let error = discover(temp.path(), &[]).unwrap_err();
        assert!(error.to_string().contains(HEADER_FILE));
    }
}
