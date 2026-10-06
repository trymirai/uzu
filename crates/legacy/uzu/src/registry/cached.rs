use std::{future::Future, pin::Pin};

use shoji::{traits::Registry, types::model::Model};
use tokio::sync::Mutex;

use crate::registry::RegistryError;

pub struct CachedRegistry {
    registry: Box<dyn Registry<Error = RegistryError>>,
    listing: Mutex<Option<(Vec<Model>, bool)>>,
}

impl CachedRegistry {
    pub fn new(registry: Box<dyn Registry<Error = RegistryError>>) -> Self {
        Self {
            registry,
            listing: Mutex::new(None),
        }
    }
}

impl Registry for CachedRegistry {
    type Error = RegistryError;

    fn identifier(&self) -> String {
        self.registry.identifier()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
        Box::pin(async { Ok(self.listing().await?.0) })
    }

    fn listing(&self) -> Pin<Box<dyn Future<Output = Result<(Vec<Model>, bool), RegistryError>> + Send + '_>> {
        Box::pin(async {
            let mut cached_listing = self.listing.lock().await;
            if let Some(cached_listing) = cached_listing.as_ref() {
                Ok(cached_listing.clone())
            } else {
                let listing = self.registry.listing().await?;
                *cached_listing = Some(listing.clone());
                Ok(listing)
            }
        })
    }
}
