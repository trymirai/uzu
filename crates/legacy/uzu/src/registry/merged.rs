use std::{future::Future, pin::Pin};

use shoji::{traits::Registry, types::model::Model};

use crate::registry::RegistryError;

pub struct MergedRegistry {
    registries: Vec<Box<dyn Registry<Error = RegistryError>>>,
}

impl MergedRegistry {
    pub fn new(registries: Vec<Box<dyn Registry<Error = RegistryError>>>) -> Self {
        Self {
            registries,
        }
    }

    pub fn add(
        &mut self,
        registry: Box<dyn Registry<Error = RegistryError>>,
    ) -> Result<(), RegistryError> {
        if self.registries.iter().any(|current_registry| current_registry.identifier() == registry.identifier()) {
            return Err(RegistryError::UnableToAddRegistry {
                identifier: registry.identifier(),
            });
        }
        self.registries.push(registry);
        Ok(())
    }

    pub fn remove(
        &mut self,
        identifier: &str,
    ) -> Result<(), RegistryError> {
        self.registries.retain(|registry| registry.identifier() != identifier);
        Ok(())
    }
}

impl Registry for MergedRegistry {
    type Error = RegistryError;

    fn identifier(&self) -> String {
        self.registries.iter().map(|registry| registry.identifier()).collect::<Vec<String>>().join(":")
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
        Box::pin(async {
            let results = futures::future::join_all(self.registries.iter().map(|registry| registry.models())).await;

            let mut models = Vec::new();
            for (registry, result) in self.registries.iter().zip(results) {
                match result {
                    Ok(registry_models) => models.extend(registry_models),
                    Err(error) => {
                        tracing::warn!(?error, registry = %registry.identifier(), "skipping registry that failed to list models");
                    },
                }
            }
            Ok(models)
        })
    }
}
