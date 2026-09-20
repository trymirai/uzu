use std::{fs, future::Future, path::Path, pin::Pin};

use backend_needle::config::{
    BACKEND_IDENTIFIER, BACKEND_NAME, Config as NeedleConfig, NEEDLE3_ENGINE_VERSION, model_identifier,
};
use shoji::{
    traits::Registry as RegistryTrait,
    types::{
        basic::Metadata,
        model::{
            Model, ModelAccessibility, ModelFamily, ModelProperties, ModelQuantization, ModelReference,
            ModelSpecialization, ModelVendor,
        },
    },
};

use crate::registry::RegistryError;

pub struct Registry {
    config: NeedleConfig,
}

impl Registry {
    pub fn new(config: NeedleConfig) -> Result<Self, RegistryError> {
        if config.cact_paths.is_empty() {
            return Err(RegistryError::UnableToCreate {
                message: "no needle3 .cact files to register".to_string(),
            });
        }
        Ok(Self {
            config,
        })
    }
}

impl RegistryTrait for Registry {
    type Error = RegistryError;

    fn identifier(&self) -> String {
        BACKEND_IDENTIFIER.to_string()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
        Box::pin(async { Ok(self.config.cact_paths.iter().filter_map(|path| self.model(path).ok()).collect()) })
    }
}

impl Registry {
    fn model(
        &self,
        path: &Path,
    ) -> Result<Model, RegistryError> {
        let identifier = model_identifier(path);
        let size =
            fs::metadata(path).map(|meta| meta.len() as i64).map_err(|error| RegistryError::UnableToGetModels {
                message: error.to_string(),
            })?;
        let vendor = ModelVendor {
            identifier: "cactus-compute".to_string(),
            metadata: Metadata {
                identifier: "cactus-compute".to_string(),
                name: "Cactus Compute".to_string(),
                description: None,
                icons: vec![],
            },
        };
        let mut model = Model::external(
            identifier.clone(),
            BACKEND_IDENTIFIER.to_string(),
            BACKEND_NAME.to_string(),
            BACKEND_IDENTIFIER.to_string(),
            BACKEND_NAME.to_string(),
            NEEDLE3_ENGINE_VERSION.to_string(),
            vec![ModelSpecialization::Chat {}],
            ModelAccessibility::Local {
                reference: ModelReference::Local {
                    path: path.to_string_lossy().to_string(),
                },
            },
            None,
        );
        model.family = Some(ModelFamily {
            identifier: "needle3".to_string(),
            vendor: vendor.clone(),
            metadata: Metadata {
                identifier: "needle3".to_string(),
                name: "Needle 3".to_string(),
                description: Some(
                    "On-device tool calling and structured extraction. Not a general chat model.".to_string(),
                ),
                icons: vec![],
            },
        });
        model.quantization = Some(ModelQuantization {
            identifier: "cactus-quants".to_string(),
            method: "cactus-quants".to_string(),
            bits_per_weight: 2,
            vendor: vendor.clone(),
            metadata: Metadata::external("Cactus Quants".to_string()),
        });
        model.properties = Some(ModelProperties {
            identifier: identifier.clone(),
            size,
            version: Some(NEEDLE3_ENGINE_VERSION.to_string()),
            metadata: Metadata::external(identifier),
        });
        Ok(model)
    }
}
