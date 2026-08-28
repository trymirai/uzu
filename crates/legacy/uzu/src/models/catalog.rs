use std::{path::PathBuf, sync::Arc};

use shoji::types::model::Model;

use super::{ModelsResolver, ResolvedModels};
use crate::{helpers::SharedAccess, registry::RegistryError};

#[derive(Clone)]
pub struct ModelCatalog {
    resolver: ModelsResolver,
    published: SharedAccess<ResolvedModels>,
}

impl ModelCatalog {
    pub async fn new(
        api_key: Option<Arc<str>>,
        cache_path: PathBuf,
    ) -> Result<Self, RegistryError> {
        let resolver = ModelsResolver::new(api_key, cache_path)?;
        let published = match resolver.load_cache().await {
            Ok(Some(models)) => models,
            Ok(None) => ResolvedModels::default(),
            Err(error) => {
                tracing::warn!(?error, "failed to load resolved models cache");
                ResolvedModels::default()
            },
        };
        Ok(Self {
            resolver,
            published: SharedAccess::new(published),
        })
    }

    pub async fn models(&self) -> Vec<Model> {
        self.published.lock().await.models()
    }

    pub async fn snapshot(&self) -> ResolvedModels {
        self.published.lock().await.clone()
    }

    pub async fn resolve(
        &self,
        models: Vec<Model>,
    ) -> Result<ResolvedModels, RegistryError> {
        let previous = self.snapshot().await;
        self.resolver.resolve(models, &previous).await
    }

    pub async fn commit(
        &self,
        models: ResolvedModels,
    ) {
        if let Err(error) = self.resolver.save_cache(&models).await {
            tracing::warn!(?error, "failed to save resolved models cache");
        }
        *self.published.lock().await = models;
    }
}
