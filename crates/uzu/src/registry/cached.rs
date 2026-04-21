use std::{future::Future, pin::Pin};

use shoji::{traits::Registry, types::model::Model};
use tokio::sync::Mutex;

use crate::registry::Error;

pub struct CachedRegistry {
    registry: Box<dyn Registry<Error = Error>>,
    models: Mutex<Option<Vec<Model>>>,
}

impl CachedRegistry {
    pub fn new(registry: Box<dyn Registry<Error = Error>>) -> Self {
        Self {
            registry,
            models: Mutex::new(None),
        }
    }

    pub async fn clear(&self) {
        let mut cached_models = self.models.lock().await;
        *cached_models = None;
    }
}

impl Registry for CachedRegistry {
    type Error = Error;

    fn indentifier(&self) -> String {
        self.registry.indentifier()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, Error>> + Send + '_>> {
        Box::pin(async {
            let mut cached_models = self.models.lock().await;
            if let Some(cached_models) = cached_models.as_ref() {
                return Ok(cached_models.clone());
            } else {
                let models = self.registry.models().await?;
                *cached_models = Some(models.clone());
                return Ok(models);
            }
        })
    }
}
