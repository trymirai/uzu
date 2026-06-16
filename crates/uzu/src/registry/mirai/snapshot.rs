use std::{
    fs::{create_dir_all, read_dir, read_to_string, rename, write},
    path::{Path, PathBuf},
};

use shoji::types::model::Model;

use crate::registry::{RegistryError, mirai::Response};

pub fn snapshot_path(cache_path: &Path) -> PathBuf {
    cache_path.join("registry").join("models.json")
}

pub fn load_snapshot(cache_path: &Path) -> Result<Response, RegistryError> {
    let path = snapshot_path(cache_path);
    let contents = read_to_string(&path).map_err(|error| RegistryError::UnableToGetModels {
        message: format!("Unable to read Mirai registry snapshot at {}: {}", path.display(), error),
    })?;
    serde_json::from_str(&contents).map_err(|error| RegistryError::UnableToGetModels {
        message: format!("Unable to parse Mirai registry snapshot at {}: {}", path.display(), error),
    })
}

pub fn save_snapshot(
    cache_path: &Path,
    response: &Response,
) -> Result<(), RegistryError> {
    let path = snapshot_path(cache_path);
    if let Some(parent) = path.parent() {
        create_dir_all(parent).map_err(|error| RegistryError::UnableToGetModels {
            message: format!("Unable to create Mirai registry snapshot directory at {}: {}", parent.display(), error),
        })?;
    }

    let tmp_path = path.with_extension("json.tmp");
    let contents = serde_json::to_vec_pretty(response).map_err(|error| RegistryError::UnableToGetModels {
        message: format!("Unable to serialize Mirai registry snapshot: {}", error),
    })?;
    write(&tmp_path, contents).map_err(|error| RegistryError::UnableToGetModels {
        message: format!("Unable to write Mirai registry snapshot at {}: {}", tmp_path.display(), error),
    })?;
    rename(&tmp_path, &path).map_err(|error| RegistryError::UnableToGetModels {
        message: format!("Unable to replace Mirai registry snapshot at {}: {}", path.display(), error),
    })
}

pub fn scan_model_metadata(cache_path: &Path) -> Vec<Model> {
    let mirai_models_path = cache_path.join("models").join("mirai");
    let Ok(entries) = read_dir(mirai_models_path) else {
        return vec![];
    };

    entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path().join("model.json");
            let contents = read_to_string(&path).ok()?;
            serde_json::from_str::<Model>(&contents)
                .map_err(|error| {
                    tracing::warn!(?error, path = %path.display(), "failed to parse cached Mirai model metadata");
                })
                .ok()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use shoji::types::{
        basic::{File, Hash, HashMethod, Metadata, Repository},
        model::{Model, ModelAccessibility, ModelBackend, ModelReference, ModelRegistry, ModelSpecialization},
    };

    use super::*;
    use crate::registry::mirai::types::{Backend as ResponseBackend, Registry as ResponseRegistry, ResponseModel};

    fn metadata(
        identifier: &str,
        name: &str,
    ) -> Metadata {
        Metadata {
            identifier: identifier.to_string(),
            name: name.to_string(),
            description: None,
            icons: vec![],
        }
    }

    fn file() -> File {
        File {
            url: "https://example.com/model.safetensors".to_string(),
            name: "model.safetensors".to_string(),
            size: 1,
            hashes: vec![Hash {
                method: HashMethod::CRC32C,
                value: "00000000".to_string(),
            }],
        }
    }

    fn model(
        identifier: &str,
        repo_id: &str,
    ) -> Model {
        Model {
            identifier: identifier.to_string(),
            registry: ModelRegistry {
                identifier: "mirai".to_string(),
                metadata: metadata("mirai-metadata", "Mirai"),
            },
            backends: vec![ModelBackend {
                identifier: "uzu".to_string(),
                version: "1.0".to_string(),
                metadata: metadata("uzu-metadata", "Uzu"),
            }],
            family: None,
            properties: None,
            quantization: None,
            specializations: vec![ModelSpecialization::Chat {}],
            accessibility: ModelAccessibility::Local {
                reference: ModelReference::Mirai {
                    toolchain_version: "1.0".to_string(),
                    repository: Some(Repository {
                        identifier: repo_id.to_string(),
                        commit_hash: None,
                        paths: None,
                    }),
                    source_repository: None,
                    files: vec![file()],
                },
            },
        }
    }

    fn response() -> Response {
        Response {
            models: vec![ResponseModel {
                id: "test-model".to_string(),
                registry: ResponseRegistry {
                    id: "mirai".to_string(),
                    metadata_id: "mirai-metadata".to_string(),
                },
                backends: vec![ResponseBackend {
                    id: "uzu".to_string(),
                    version: "1.0".to_string(),
                    metadata_id: "uzu-metadata".to_string(),
                }],
                family: None,
                properties: None,
                quantization: None,
                specializations: vec![ModelSpecialization::Chat {}],
                accessibility: ModelAccessibility::Local {
                    reference: ModelReference::Mirai {
                        toolchain_version: "1.0".to_string(),
                        repository: Some(Repository {
                            identifier: "trymirai/test-model".to_string(),
                            commit_hash: None,
                            paths: None,
                        }),
                        source_repository: None,
                        files: vec![file()],
                    },
                },
            }],
            metadatas: vec![metadata("mirai-metadata", "Mirai"), metadata("uzu-metadata", "Uzu")],
        }
    }

    #[test]
    fn test_save_load_snapshot_roundtrip() {
        let temp_dir = tempfile::tempdir().expect("temp dir must be created");
        let response = response();

        save_snapshot(temp_dir.path(), &response).expect("snapshot must save");
        let loaded = load_snapshot(temp_dir.path()).expect("snapshot must load");

        assert_eq!(loaded, response);
        assert_eq!(loaded.models().expect("response must convert")[0].identifier, "test-model");
    }

    #[test]
    fn test_scan_model_metadata_loads_cached_mirai_models() {
        let temp_dir = tempfile::tempdir().expect("temp dir must be created");
        let model = model("test-model", "trymirai/test-model");
        let model_dir = temp_dir.path().join("models").join("mirai").join(model.cache_identifier());
        std::fs::create_dir_all(&model_dir).expect("model dir must be created");
        std::fs::write(model_dir.join("model.json"), serde_json::to_vec_pretty(&model).expect("model must serialize"))
            .expect("model metadata must be written");

        let models = scan_model_metadata(temp_dir.path());

        assert_eq!(models, vec![model]);
    }
}
