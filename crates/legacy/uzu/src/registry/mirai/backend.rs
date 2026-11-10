use serde::{Deserialize, Serialize};
use shoji::types::model::{Model, ModelAccessibility, ModelSource};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Backend {
    pub identifier: String,
    pub version: String,
}

impl Backend {
    pub(crate) fn supports_model(
        &self,
        model: &Model,
    ) -> bool {
        // Engine selects the first advertised backend when loading a model.
        let Some(backend) = model.backends.first() else {
            return false;
        };
        if backend.identifier != self.identifier || backend.version != self.version {
            return false;
        }
        match &model.accessibility {
            ModelAccessibility::OnDevice {
                source:
                    ModelSource::Registry {
                        toolchain_version,
                        ..
                    },
            } => toolchain_version == &self.version,
            _ => true,
        }
    }
}
