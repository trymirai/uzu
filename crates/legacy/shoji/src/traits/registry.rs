use std::{error::Error, future::Future, pin::Pin, sync::Arc};

use crate::types::model::{Model, ModelIdentifier};

pub trait Registry: Send + Sync {
    type Error: Error;

    fn identifier(&self) -> String;

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, Self::Error>> + Send + '_>>;

    fn listing(&self) -> Pin<Box<dyn Future<Output = Result<(Vec<Model>, bool), Self::Error>> + Send + '_>> {
        Box::pin(async { Ok((self.models().await?, true)) })
    }

    /// A snapshot that can be read without waiting for network requests.
    /// `None` means no listing has loaded; an empty listing is still a snapshot.
    fn cached_listing(&self) -> Option<(Vec<Model>, bool)> {
        None
    }

    /// Refreshes the snapshot, notifying after intermediate results become readable.
    fn refresh_listing(
        &self,
        _on_update: Arc<dyn Fn() + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = Result<(Vec<Model>, bool), Self::Error>> + Send + '_>> {
        self.listing()
    }

    fn model_by_identifier(
        &self,
        identifier: &ModelIdentifier,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Model>, Self::Error>> + Send + '_>> {
        let identifier = identifier.clone();
        Box::pin(async move {
            let models = self.models().await?;
            let model = models.iter().find(|model| model.identifier == identifier).cloned();
            Ok(model)
        })
    }

    fn model_by_repo_id(
        &self,
        repo_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Model>, Self::Error>> + Send + '_>> {
        let repo_id = repo_id.to_string();
        Box::pin(async move {
            let models = self.models().await?;
            let model = models.iter().find(|model| model.repo_ids().contains(&repo_id)).cloned();
            Ok(model)
        })
    }
}
