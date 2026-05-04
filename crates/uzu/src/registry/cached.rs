use std::{future::Future, pin::Pin};

use shoji::{traits::Registry, types::model::Model};
use tokio::sync::Mutex;

use crate::registry::RegistryError;

pub struct CachedRegistry {
    registry: Box<dyn Registry<Error = RegistryError>>,
    models: Mutex<Option<Vec<Model>>>,
}

impl CachedRegistry {
    pub fn new(registry: Box<dyn Registry<Error = RegistryError>>) -> Self {
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
    type Error = RegistryError;

    fn indentifier(&self) -> String {
        self.registry.indentifier()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
        Box::pin(async {
            let mut cached_models = self.models.lock().await;
            if let Some(cached_models) = cached_models.as_ref() {
                return Ok(cached_models.clone());
            } else {
                let models = match self.registry.models().await {
                    Ok(models) => models.clone(),
                    Err(_) => {
                        vec![]
                    },
                };
                *cached_models = Some(models.clone());
                return Ok(models);
            }
        })
    }
}
