use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
};

use shoji::{traits::Registry, types::model::Model};

use crate::registry::RegistryError;

pub struct CachedRegistry {
    registry: Arc<dyn Registry<Error = RegistryError>>,
    listing: Arc<Mutex<Option<(Vec<Model>, bool)>>>,
    refresh: tokio::sync::Mutex<()>,
}

impl CachedRegistry {
    pub fn new(registry: Box<dyn Registry<Error = RegistryError>>) -> Self {
        Self {
            listing: Arc::new(Mutex::new(registry.cached_listing())),
            registry: registry.into(),
            refresh: tokio::sync::Mutex::new(()),
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
            if let Some(listing) = self.cached_listing() {
                return Ok(listing);
            }
            self.refresh_listing(Arc::new(|| {})).await
        })
    }

    fn cached_listing(&self) -> Option<(Vec<Model>, bool)> {
        self.listing.lock().expect("registry snapshot mutex poisoned").clone()
    }

    fn refresh_listing(
        &self,
        on_update: Arc<dyn Fn() + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = Result<(Vec<Model>, bool), RegistryError>> + Send + '_>> {
        Box::pin(async move {
            let _refresh = self.refresh.lock().await;
            let had_snapshot = {
                let mut snapshot = self.listing.lock().expect("registry snapshot mutex poisoned");
                if let Some((_, complete)) = snapshot.as_mut() {
                    *complete = false;
                }
                snapshot.is_some()
            };
            if had_snapshot {
                on_update();
            }
            let registry = self.registry.clone();
            let snapshot = self.listing.clone();
            let notify = on_update.clone();
            let listing = self
                .registry
                .refresh_listing(Arc::new(move || {
                    if let Some(listing) = registry.cached_listing() {
                        *snapshot.lock().expect("registry snapshot mutex poisoned") = Some(listing);
                        notify();
                    }
                }))
                .await?;
            *self.listing.lock().expect("registry snapshot mutex poisoned") = Some(listing.clone());
            on_update();
            Ok(listing)
        })
    }
}
