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
                Ok(cached_models.clone())
            } else {
                let models = self.registry.models().await?;
                *cached_models = Some(models.clone());
                Ok(models)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{future::Future, pin::Pin, sync::Arc};

    use tokio::sync::Mutex;

    use super::*;

    struct FlakyRegistry {
        calls: Arc<Mutex<usize>>,
    }

    impl Registry for FlakyRegistry {
        type Error = RegistryError;

        fn indentifier(&self) -> String {
            "flaky".to_string()
        }

        fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
            Box::pin(async {
                let mut calls = self.calls.lock().await;
                *calls += 1;
                if *calls == 1 {
                    Err(RegistryError::UnableToGetModels {
                        message: "temporary failure".to_string(),
                    })
                } else {
                    Ok(vec![])
                }
            })
        }
    }

    #[tokio::test]
    async fn test_cached_registry_does_not_cache_errors() {
        let calls = Arc::new(Mutex::new(0));
        let registry = CachedRegistry::new(Box::new(FlakyRegistry {
            calls: calls.clone(),
        }));

        let first_result = registry.models().await;
        let second_result = registry.models().await;

        assert!(first_result.is_err());
        assert!(second_result.is_ok());
        assert_eq!(*calls.lock().await, 2);
    }
}
