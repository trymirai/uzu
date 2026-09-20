use std::{
    env, fs,
    path::{Path, PathBuf},
};

use crate::error::Error;

pub const NEEDLE3_ENGINE_VERSION: &str = "3.0.1";
pub const BACKEND_IDENTIFIER: &str = "needle";
pub const BACKEND_NAME: &str = "Needle";
pub const DEFAULT_MODEL_IDENTIFIER: &str = "cactus:needle3";

pub const NEEDLE2_TAG: u32 = 0x05E12A83;
pub const NEEDLE3_TAG: u32 = 0x05E12A84;

pub const KEY_NEEDLE3_LIB_PATH: &str = "NEEDLE3_LIB_PATH";
pub const KEY_NEEDLE_LIB_PATH: &str = "NEEDLE_LIB_PATH";
pub const KEY_NEEDLE_WEIGHTS_PATH: &str = "NEEDLE_WEIGHTS_PATH";
pub const KEY_NEEDLE_MODELS_DIR: &str = "NEEDLE_MODELS_DIR";

const BASE_WEIGHTS: &str = "needle3.cact";

#[derive(Debug, Clone, Default)]
pub struct DiscoveryHints {
    pub lib_path: Option<String>,
    pub weights_path: Option<String>,
    pub models_dir: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub lib_path: PathBuf,
    pub cact_paths: Vec<PathBuf>,
}

impl Config {
    pub fn discover_from_env() -> Result<Self, Error> {
        Self::discover(DiscoveryHints {
            lib_path: env::var(KEY_NEEDLE3_LIB_PATH).ok(),
            weights_path: env::var(KEY_NEEDLE_WEIGHTS_PATH).ok(),
            models_dir: env::var(KEY_NEEDLE_MODELS_DIR).ok(),
        })
    }

    pub fn discover(hints: DiscoveryHints) -> Result<Self, Error> {
        let mut lib_tried = Vec::new();
        let mut cact_tried = Vec::new();

        push_unique(&mut lib_tried, hints.lib_path.as_deref().map(PathBuf::from));
        if hints.lib_path.is_none() {
            push_unique(&mut lib_tried, env::var(KEY_NEEDLE3_LIB_PATH).ok().map(PathBuf::from));
        }
        if let Ok(legacy) = env::var(KEY_NEEDLE_LIB_PATH) {
            tracing::warn!(
                path = %legacy,
                "NEEDLE_LIB_PATH is a Needle 2 fallback; prefer NEEDLE3_LIB_PATH"
            );
            push_unique(&mut lib_tried, Some(PathBuf::from(legacy)));
        }
        push_unique(&mut lib_tried, Some(cache_dir().join(lib_name())));

        push_unique(&mut cact_tried, hints.weights_path.as_deref().map(PathBuf::from));
        if hints.weights_path.is_none() {
            push_unique(&mut cact_tried, env::var(KEY_NEEDLE_WEIGHTS_PATH).ok().map(PathBuf::from));
        }
        push_unique(&mut cact_tried, Some(cache_dir().join(BASE_WEIGHTS)));

        let models_dir = hints.models_dir.clone().or_else(|| env::var(KEY_NEEDLE_MODELS_DIR).ok());
        if let Some(dir) = models_dir {
            push_cact_dir(&mut cact_tried, Path::new(&dir));
        }

        let existing_cacts: Vec<PathBuf> = cact_tried.iter().filter(|path| path.is_file()).cloned().collect();
        for cact in &existing_cacts {
            if let Some(parent) = cact.parent() {
                push_unique(&mut lib_tried, Some(parent.join(lib_name())));
            }
        }
        let existing_libs: Vec<PathBuf> = lib_tried.iter().filter(|path| path.is_file()).cloned().collect();
        for lib in &existing_libs {
            if let Some(parent) = lib.parent() {
                push_unique(&mut cact_tried, Some(parent.join(BASE_WEIGHTS)));
            }
        }

        let lib_existing: Vec<PathBuf> = lib_tried.iter().filter(|path| path.is_file()).cloned().collect();
        if lib_existing.is_empty() {
            return Err(Error::LibraryNotFound {
                tried: lib_tried,
            });
        }

        let mut cact_existing = Vec::new();
        for path in cact_tried.iter().filter(|path| path.is_file()) {
            match read_cact_tag(path) {
                Ok(NEEDLE3_TAG) => push_unique(&mut cact_existing, Some(path.clone())),
                Ok(tag) => {
                    tracing::warn!(path = %path.display(), tag, "skipping non-Needle-3 archive");
                },
                Err(error) => {
                    tracing::warn!(path = %path.display(), %error, "skipping unreadable archive");
                },
            }
        }
        if cact_existing.is_empty() {
            return Err(Error::WeightsNotFound {
                tried: cact_tried,
            });
        }

        Ok(Self {
            lib_path: lib_existing[0].clone(),
            cact_paths: cact_existing,
        })
    }
}

pub fn model_identifier(path: &Path) -> String {
    let stem = path.file_stem().and_then(|stem| stem.to_str()).unwrap_or("needle3");
    if stem == "needle3" {
        DEFAULT_MODEL_IDENTIFIER.to_string()
    } else {
        format!("cactus:needle3:{stem}")
    }
}

pub fn read_cact_tag(path: &Path) -> Result<u32, Error> {
    let bytes = fs::read(path).map_err(|error| Error::ReadFailed {
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    if bytes.len() < 4 {
        return Err(Error::UnsupportedGeneration {
            path: path.to_path_buf(),
            tag: 0,
        });
    }
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

pub fn require_needle3(path: &Path) -> Result<(), Error> {
    let tag = read_cact_tag(path)?;
    if tag == NEEDLE3_TAG {
        Ok(())
    } else {
        Err(Error::UnsupportedGeneration {
            path: path.to_path_buf(),
            tag,
        })
    }
}

fn cache_dir() -> PathBuf {
    home_dir().join(".cache").join("cactus-needle").join("v3").join(NEEDLE3_ENGINE_VERSION)
}

fn home_dir() -> PathBuf {
    env::var_os("HOME").or_else(|| env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."))
}

fn lib_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libneedle.dylib"
    } else if cfg!(target_os = "windows") {
        "libneedle.dll"
    } else {
        "libneedle.so"
    }
}

fn push_unique(
    paths: &mut Vec<PathBuf>,
    path: Option<PathBuf>,
) {
    let Some(path) = path else {
        return;
    };
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn push_cact_dir(
    paths: &mut Vec<PathBuf>,
    dir: &Path,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("cact") {
            push_unique(paths, Some(path));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_needle3(path: &Path) {
        fs::write(path, NEEDLE3_TAG.to_le_bytes()).unwrap();
    }

    #[test]
    fn model_identifier_default_and_custom_stems() {
        assert_eq!(model_identifier(Path::new("/tmp/needle3.cact")), DEFAULT_MODEL_IDENTIFIER);
        assert_eq!(model_identifier(Path::new("/tmp/invoice.cact")), "cactus:needle3:invoice");
    }

    #[test]
    fn require_needle3_rejects_other_tags() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("needle2.cact");
        fs::write(&path, NEEDLE2_TAG.to_le_bytes()).unwrap();
        let error = require_needle3(&path).unwrap_err();
        assert!(matches!(error, Error::UnsupportedGeneration { tag, .. } if tag == NEEDLE2_TAG));
    }

    #[test]
    fn discover_finds_sibling_lib_and_cact() {
        let dir = tempfile::tempdir().unwrap();
        let lib = dir.path().join(lib_name());
        let cact = dir.path().join(BASE_WEIGHTS);
        fs::write(&lib, []).unwrap();
        write_needle3(&cact);

        let config = Config::discover(DiscoveryHints {
            lib_path: None,
            weights_path: Some(cact.to_string_lossy().into_owned()),
            models_dir: None,
        })
        .unwrap();
        assert_eq!(config.lib_path, lib);
        assert_eq!(config.cact_paths, vec![cact]);
    }

    #[test]
    fn discover_lists_cact_files_from_models_dir() {
        let dir = tempfile::tempdir().unwrap();
        let lib = dir.path().join(lib_name());
        fs::write(&lib, []).unwrap();
        let first = dir.path().join(BASE_WEIGHTS);
        let second = dir.path().join("invoice.cact");
        write_needle3(&first);
        write_needle3(&second);

        let config = Config::discover(DiscoveryHints {
            lib_path: Some(lib.to_string_lossy().into_owned()),
            weights_path: None,
            models_dir: Some(dir.path().to_string_lossy().into_owned()),
        })
        .unwrap();
        assert_eq!(config.lib_path, lib);
        assert!(config.cact_paths.contains(&first));
        assert!(config.cact_paths.contains(&second));
    }
}
