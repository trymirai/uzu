use serde::{Deserialize, Serialize};

use crate::types::{
    basic::{Metadata, Value},
    model::{
        ModelAccessibility, ModelBackend, ModelFamily, ModelIdentifier, ModelProperties, ModelQuantization,
        ModelRegistry, ModelSource, ModelSpecialization,
    },
};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Model {
    #[serde(rename = "id")]
    pub identifier: ModelIdentifier,
    pub registry: ModelRegistry,
    pub backends: Vec<ModelBackend>,
    pub family: Option<ModelFamily>,
    pub properties: Option<ModelProperties>,
    pub quantization: Option<ModelQuantization>,
    pub specializations: Vec<ModelSpecialization>,
    pub accessibility: ModelAccessibility,
    #[serde(default)]
    pub encoding: Option<Value>,
}

#[bindings::export(Implementation)]
impl Model {
    #[bindings::export(Method(Getter))]
    pub fn name(&self) -> String {
        let parts: Vec<Option<String>> = vec![
            self.family.as_ref().map(|family| family.name()),
            self.properties.as_ref().map(|properties| properties.name()),
            self.quantization.as_ref().map(|quantization| quantization.name()),
        ];
        let name = parts.iter().filter_map(|part| part.clone()).collect::<Vec<_>>().join(" ");
        if name.is_empty() {
            self.identifier.clone()
        } else {
            name
        }
    }

    #[bindings::export(Method(Getter))]
    pub fn is_on_device(&self) -> bool {
        matches!(self.accessibility, ModelAccessibility::OnDevice { .. })
    }

    #[bindings::export(Method(Getter))]
    pub fn is_remote(&self) -> bool {
        matches!(self.accessibility, ModelAccessibility::Remote { .. })
    }

    #[bindings::export(Method(Getter))]
    pub fn is_downloadable(&self) -> bool {
        matches!(
            self.accessibility,
            ModelAccessibility::OnDevice {
                source: ModelSource::Registry { .. }
            }
        )
    }

    #[bindings::export(Method(Getter))]
    pub fn is_quantized(&self) -> bool {
        self.quantization.is_some()
    }

    #[bindings::export(Method(Getter))]
    pub fn cache_identifier(&self) -> String {
        self.identifier.replace(":", "-").replace("/", "-")
    }

    #[bindings::export(Method(Getter))]
    pub fn repo_ids(&self) -> Vec<String> {
        match &self.accessibility {
            ModelAccessibility::OnDevice {
                source:
                    ModelSource::Registry {
                        repository,
                        source_repository,
                        ..
                    },
            } => repository
                .iter()
                .chain(source_repository.iter())
                .map(|repository| repository.identifier.clone())
                .collect(),
            ModelAccessibility::OnDevice {
                source: ModelSource::Filesystem {
                    ..
                },
            } => vec![],
            ModelAccessibility::Remote {
                repository,
            } => repository.iter().map(|repository| repository.identifier.clone()).collect(),
        }
    }

    #[bindings::export(Method(Getter))]
    pub fn filesystem_path(&self) -> Option<String> {
        match &self.accessibility {
            ModelAccessibility::OnDevice {
                source: ModelSource::Filesystem {
                    path,
                },
            } => Some(path.clone()),
            _ => None,
        }
    }

    #[bindings::export(Method(Getter))]
    pub fn checkpoint_version(&self) -> Option<String> {
        match &self.accessibility {
            ModelAccessibility::OnDevice {
                source:
                    ModelSource::Registry {
                        toolchain_version,
                        repository,
                        ..
                    },
            } => repository
                .as_ref()
                .and_then(|repository| repository.commit_hash.clone())
                .or_else(|| Some(toolchain_version.clone())),
            _ => None,
        }
    }
}

#[bindings::export(Implementation)]
impl Model {
    #[bindings::export(Method(Factory))]
    pub fn external(
        identifier: ModelIdentifier,
        registry_identifier: String,
        registry_name: String,
        backend_identifier: String,
        backend_name: String,
        backend_version: String,
        specializations: Vec<ModelSpecialization>,
        accessibility: ModelAccessibility,
        encoding: Option<Value>,
    ) -> Self {
        let registry = ModelRegistry {
            identifier: registry_identifier,
            metadata: Metadata::external(registry_name),
        };
        let backend = ModelBackend {
            identifier: backend_identifier.clone(),
            version: backend_version,
            metadata: Metadata::external(backend_name),
        };
        Self {
            identifier,
            registry,
            backends: vec![backend],
            family: None,
            properties: None,
            quantization: None,
            specializations,
            accessibility,
            encoding,
        }
    }
}

#[bindings::export(Implementation)]
impl Model {
    #[bindings::export(Method(Getter))]
    pub fn is_chat_capable(&self) -> bool {
        self.specializations.contains(&ModelSpecialization::Chat {})
    }

    #[bindings::export(Method(Getter))]
    pub fn is_classification_capable(&self) -> bool {
        self.specializations.contains(&ModelSpecialization::Classification {})
    }

    #[bindings::export(Method(Getter))]
    pub fn is_text_to_speech_capable(&self) -> bool {
        self.specializations.contains(&ModelSpecialization::TextToSpeech {})
    }

    #[bindings::export(Method(Getter))]
    pub fn is_translation_capable(&self) -> bool {
        self.specializations.contains(&ModelSpecialization::Translation {})
    }

    #[bindings::export(Method(Getter))]
    pub fn is_speculation_capable(&self) -> bool {
        self.specializations.contains(&ModelSpecialization::Speculation {})
    }
}
