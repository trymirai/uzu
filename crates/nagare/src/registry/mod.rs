mod cached;
mod error;
mod fixed;
mod merged;
pub mod types;

use std::{future::Future, pin::Pin};

pub use cached::CachedRegistry;
pub use error::Error;
pub use fixed::FixedRegistry;
pub use merged::MergedRegistry;

use crate::registry::types::Model;

pub trait Registry: Send + Sync {
    fn indentifier(&self) -> String;

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, Error>> + Send + '_>>;

    fn model_by_identifier(
        &self,
        identifier: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Model>, Error>> + Send + '_>> {
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
    ) -> Pin<Box<dyn Future<Output = Result<Option<Model>, Error>> + Send + '_>> {
        let repo_id = repo_id.to_string();
        Box::pin(async move {
            let models = self.models().await?;
            let model = models.iter().find(|model| model.repo_ids().contains(&repo_id)).cloned();
            Ok(model)
        })
    }
}
