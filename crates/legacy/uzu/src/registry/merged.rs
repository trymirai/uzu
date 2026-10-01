use std::{future::Future, pin::Pin};

use shoji::{
    traits::Registry,
    types::model::{Model, ModelIdentifier},
};

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

    pub async fn model(
        &self,
        identifier: &str,
    ) -> Result<Option<Model>, RegistryError> {
        unique_model(
            identifier,
            self.models().await?.into_iter().filter(|model| {
                model.identifier == identifier || model.repo_ids().iter().any(|repo_id| repo_id == identifier)
            }),
        )
    }
}

pub(crate) fn unique_model(
    identifier: &str,
    mut models: impl Iterator<Item = Model>,
) -> Result<Option<Model>, RegistryError> {
    let model = models.next();
    if models.next().is_some() {
        return Err(RegistryError::UnableToGetModels {
            message: format!("Ambiguous model reference `{identifier}`: matches multiple models"),
        });
    }
    Ok(model)
}

impl Registry for MergedRegistry {
    type Error = RegistryError;

    fn identifier(&self) -> String {
        self.registries.iter().map(|registry| registry.identifier()).collect::<Vec<String>>().join(":")
    }

    fn model_by_identifier(
        &self,
        identifier: &ModelIdentifier,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Model>, RegistryError>> + Send + '_>> {
        let identifier = identifier.clone();
        Box::pin(async move {
            unique_model(&identifier, self.models().await?.into_iter().filter(|model| model.identifier == identifier))
        })
    }

    fn model_by_repo_id(
        &self,
        repo_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Model>, RegistryError>> + Send + '_>> {
        let repo_id = repo_id.to_string();
        Box::pin(async move {
            unique_model(&repo_id, self.models().await?.into_iter().filter(|model| model.repo_ids().contains(&repo_id)))
        })
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
