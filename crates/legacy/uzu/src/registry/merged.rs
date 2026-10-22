use std::{future::Future, pin::Pin, sync::Arc};

use shoji::{
    traits::Registry,
    types::model::{Model, ModelIdentifier},
};

use crate::registry::RegistryError;

#[derive(Clone)]
pub struct MergedRegistry {
    registries: Vec<Arc<dyn Registry<Error = RegistryError>>>,
}

impl MergedRegistry {
    pub fn new(registries: Vec<Box<dyn Registry<Error = RegistryError>>>) -> Self {
        Self {
            registries: registries.into_iter().map(Arc::from).collect(),
        }
    }

    pub fn add(
        &mut self,
        registry: Arc<dyn Registry<Error = RegistryError>>,
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
        Box::pin(async { Ok(self.listing().await?.0) })
    }

    fn listing(&self) -> Pin<Box<dyn Future<Output = Result<(Vec<Model>, bool), RegistryError>> + Send + '_>> {
        Box::pin(async {
            if let Some(listing) = self.cached_listing() {
                return Ok(listing);
            }
            self.refresh_listing(Arc::new(|| {})).await
        })
    }

    fn cached_listing(&self) -> Option<(Vec<Model>, bool)> {
        let mut models = Vec::new();
        let mut complete = true;
        let mut loaded = self.registries.is_empty();
        for registry in &self.registries {
            match registry.cached_listing() {
                Some((snapshot, registry_complete)) => {
                    loaded = true;
                    models.extend(snapshot);
                    complete &= registry_complete;
                },
                None => complete = false,
            }
        }
        loaded.then_some((models, complete))
    }

    fn refresh_listing(
        &self,
        on_update: Arc<dyn Fn() + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = Result<(Vec<Model>, bool), RegistryError>> + Send + '_>> {
        Box::pin(async move {
            let results = futures::future::join_all(
                self.registries.iter().map(|registry| registry.refresh_listing(on_update.clone())),
            )
            .await;

            let mut models = Vec::new();
            let mut complete = true;
            for (registry, result) in self.registries.iter().zip(results) {
                match result {
                    Ok((registry_models, registry_complete)) => {
                        models.extend(registry_models);
                        complete &= registry_complete;
                    },
                    Err(error) => {
                        complete = false;
                        if let Some((cached, _)) = registry.cached_listing() {
                            models.extend(cached);
                        }
                        tracing::warn!(?error, registry = %registry.identifier(), "skipping registry that failed to list models");
                    },
                }
            }
            Ok((models, complete))
        })
    }
}
