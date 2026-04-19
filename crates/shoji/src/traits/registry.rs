use std::{error::Error, future::Future, pin::Pin};

use crate::types::Model;

pub trait Registry: Send + Sync {
    type Error: Error;

    fn indentifier(&self) -> String;

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, Self::Error>> + Send + '_>>;

    fn model_by_identifier(
        &self,
        identifier: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Model>, Self::Error>> + Send + '_>> {
        let identifier = identifier.to_string();
        Box::pin(async move {
            let models = self.models().await?;
            let model = models.iter().find(|model| model.identifier() == identifier).cloned();
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
