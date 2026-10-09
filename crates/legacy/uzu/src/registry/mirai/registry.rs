use std::{
    fs::{read_to_string, remove_file, rename, write},
    future::Future,
    io,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, Mutex},
};

use bon::bon;
use download_manager::BearerToken;
use futures::{StreamExt, stream};
use nagare::api::{Client, Error as ApiError};
use shoji::{
    traits::Registry as RegistryTrait,
    types::model::{Model, ModelAccessibility, ModelSource},
};

use super::{
    api::{HUGGING_FACE_URL, REGISTRY_URL},
    backend::Backend,
    fetch_models::{CatalogDevice, FetchModels},
    hugging_face::{HuggingFace, is_lower_hex},
};
use crate::{device::Device, registry::RegistryError};

const CONCURRENT_RESOLUTIONS: usize = 8;

pub struct Registry {
    device: CatalogDevice,
    backends: Vec<Backend>,
    include_traces: bool,
    cache_path: PathBuf,
    client: Client,
    hugging_face: HuggingFace,
    snapshot: Mutex<Option<(Vec<Model>, bool)>>,
}

#[bon]
impl Registry {
    #[builder]
    pub fn new(
        api_key: Option<String>,
        huggingface_api_key: Option<BearerToken>,
        device: Device,
        backends: Vec<Backend>,
        #[builder(default)] include_traces: bool,
        cache_path: PathBuf,
        #[builder(default = REGISTRY_URL.to_string(), into)] registry_url: String,
        #[builder(default = HUGGING_FACE_URL.to_string(), into)] hugging_face_url: String,
    ) -> Result<Self, RegistryError> {
        let client = Client::builder().base_url(registry_url).maybe_bearer_token(api_key).build().map_err(|error| {
            RegistryError::UnableToCreate {
                message: error.to_string(),
            }
        })?;
        let hugging_face =
            HuggingFace::builder().endpoint(hugging_face_url).maybe_token(huggingface_api_key).build()?;
        let snapshot = Self::load_registry(&cache_path).ok().map(|models| (models, false));

        Ok(Self {
            device: (&device).into(),
            backends,
            include_traces,
            cache_path,
            client,
            hugging_face,
            snapshot: Mutex::new(snapshot),
        })
    }
}

impl RegistryTrait for Registry {
    type Error = RegistryError;

    fn identifier(&self) -> String {
        "mirai".to_string()
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
        self.snapshot.lock().expect("Mirai registry snapshot mutex poisoned").clone()
    }

    fn refresh_listing(
        &self,
        on_update: Arc<dyn Fn() + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = Result<(Vec<Model>, bool), RegistryError>> + Send + '_>> {
        Box::pin(async move {
            let cached = self.cached_listing().map(|(models, _)| models).unwrap_or_default();
            if !cached.is_empty() {
                self.publish((cached.clone(), false), &on_update);
            }
            let models = self.fetch_models().await.map_err(|error| RegistryError::UnableToGetModels {
                message: error.to_string(),
            })?;
            let listing = self.resolve(models, cached, &on_update).await;
            if let Err(error) = self.save_registry(&listing.0) {
                tracing::warn!(?error, "failed to save Mirai registry");
            }
            self.publish(listing.clone(), &on_update);
            Ok(listing)
        })
    }
}

impl Registry {
    async fn fetch_models(&self) -> Result<Vec<Model>, ApiError> {
        let request = FetchModels::builder()
            .device(self.device.clone())
            .backends(self.backends.clone())
            .include_traces(self.include_traces)
            .show_all(std::env::var("UZU_REGISTRY_SHOW_ALL").is_ok())
            .build();
        let response = self.client.call::<FetchModels>(&request).await?;
        response.models().ok_or_else(|| ApiError::Decode("response contained no models".to_string()))
    }

    async fn resolve(
        &self,
        models: Vec<Model>,
        cached: Vec<Model>,
        on_update: &Arc<dyn Fn() + Send + Sync>,
    ) -> (Vec<Model>, bool) {
        // Keep the last usable descriptor while a new revision's files resolve.
        // A failed lookup must not remove another already downloaded model.
        let mut resolved: Vec<Option<Model>> = models
            .iter()
            .map(|model| cached.iter().find(|previous| previous.identifier == model.identifier).cloned())
            .collect();
        let snapshot = |resolved: &[Option<Model>]| resolved.iter().flatten().cloned().collect::<Vec<_>>();
        self.publish((snapshot(&resolved), false), on_update);
        let cached = &cached;
        let mut pending = stream::iter(models.into_iter().enumerate())
            .map(|(index, model)| async move { (index, self.resolve_model(model, cached).await) })
            .buffer_unordered(CONCURRENT_RESOLUTIONS);
        let mut complete = true;
        while let Some((index, model)) = pending.next().await {
            if let Some(model) = model {
                resolved[index] = Some(model);
                self.publish((snapshot(&resolved), false), on_update);
            } else {
                complete = false;
            }
        }
        (snapshot(&resolved), complete)
    }

    fn publish(
        &self,
        listing: (Vec<Model>, bool),
        on_update: &Arc<dyn Fn() + Send + Sync>,
    ) {
        *self.snapshot.lock().expect("Mirai registry snapshot mutex poisoned") = Some(listing);
        on_update();
    }

    async fn resolve_model(
        &self,
        mut model: Model,
        cached: &[Model],
    ) -> Option<Model> {
        let ModelAccessibility::OnDevice {
            source:
                ModelSource::Registry {
                    repository: Some(repository),
                    files,
                    ..
                },
        } = &mut model.accessibility
        else {
            return Some(model);
        };
        if repository.commit_hash.is_none() {
            return Some(model);
        }
        let pinned_to_commit = repository.commit_hash.as_deref().is_some_and(|revision| is_lower_hex(revision, 40));
        let previous = cached.iter().filter(|_| pinned_to_commit).find_map(|previous| match &previous.accessibility {
            ModelAccessibility::OnDevice {
                source:
                    ModelSource::Registry {
                        repository: Some(previous_repository),
                        files,
                        ..
                    },
            } if previous_repository == repository && self.hugging_face.serves(files) => Some(files.clone()),
            _ => None,
        });
        *files = match previous {
            Some(files) => files,
            None => match self.hugging_face.files(repository).await {
                Ok(files) => files,
                Err(error) => {
                    tracing::warn!(?error, model = %model.identifier, "skipping model with unresolved Hugging Face files");
                    return None;
                },
            },
        };
        Some(model)
    }

    fn save_registry(
        &self,
        models: &[Model],
    ) -> io::Result<()> {
        let temporary = self.cache_path.join(format!(".registry-{}.json", uuid::Uuid::new_v4()));
        let saved = (|| {
            let contents = serde_json::to_vec_pretty(models).map_err(io::Error::other)?;
            write(&temporary, contents)?;
            rename(&temporary, self.cache_path.join("registry.json"))
        })();
        if saved.is_err() {
            let _ = remove_file(temporary);
        }
        saved
    }

    fn load_registry(cache_path: &Path) -> Result<Vec<Model>, RegistryError> {
        let contents =
            read_to_string(cache_path.join("registry.json")).map_err(|error| RegistryError::UnableToGetModels {
                message: format!("Unable to read registry: {}", error),
            })?;
        serde_json::from_str(&contents).map_err(|error| RegistryError::UnableToGetModels {
            message: format!("Unable to parse registry: {}", error),
        })
    }
}
