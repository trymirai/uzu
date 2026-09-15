use std::{
    fs::{read_to_string, write},
    future::{Future, ready},
    io,
    path::PathBuf,
    pin::Pin,
};

use bon::bon;
use download_manager::BearerToken;
use futures::{StreamExt, stream};
use nagare::api::{Client, Error as ApiError, IsTransient};
use shoji::{
    traits::Registry as RegistryTrait,
    types::model::{Model, ModelAccessibility, ModelSource},
};

use super::{
    api::{HUGGING_FACE_URL, REGISTRY_URL},
    backend::Backend,
    fetch_models::FetchModels,
    hugging_face::{HuggingFace, is_lower_hex},
};
use crate::{device::Device, registry::RegistryError};

const CONCURRENT_RESOLUTIONS: usize = 8;

pub struct Registry {
    device: Device,
    backends: Vec<Backend>,
    include_traces: bool,
    cache_path: PathBuf,
    client: Client,
    hugging_face: HuggingFace,
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

        Ok(Self {
            device,
            backends,
            include_traces,
            cache_path,
            client,
            hugging_face,
        })
    }
}

impl RegistryTrait for Registry {
    type Error = RegistryError;

    fn identifier(&self) -> String {
        "mirai".to_string()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
        Box::pin(async {
            let cached = self.load_registry();
            match self.fetch_models().await {
                Ok(models) => {
                    let models = self.resolve(models, cached.unwrap_or_default()).await;
                    let saved = serde_json::to_vec_pretty(&models)
                        .map_err(io::Error::other)
                        .and_then(|contents| write(self.registry_path(), contents));
                    if let Err(error) = saved {
                        tracing::warn!(?error, "failed to save Mirai registry");
                    }
                    Ok(models)
                },
                Err(error) => {
                    if error.is_transient()
                        && let Ok(models) = cached
                    {
                        tracing::warn!(?error, "serving cached Mirai registry after fetch failure");
                        return Ok(models);
                    }
                    Err(RegistryError::UnableToGetModels {
                        message: error.to_string(),
                    })
                },
            }
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
    ) -> Vec<Model> {
        stream::iter(models)
            .map(|model| self.resolve_model(model, &cached))
            .buffered(CONCURRENT_RESOLUTIONS)
            .filter_map(ready)
            .collect()
            .await
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
            } if previous_repository == repository => Some(files.clone()),
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

    fn registry_path(&self) -> PathBuf {
        self.cache_path.join("registry.json")
    }

    fn load_registry(&self) -> Result<Vec<Model>, RegistryError> {
        let contents = read_to_string(self.registry_path()).map_err(|error| RegistryError::UnableToGetModels {
            message: format!("Unable to read registry: {}", error),
        })?;
        serde_json::from_str(&contents).map_err(|error| RegistryError::UnableToGetModels {
            message: format!("Unable to parse registry: {}", error),
        })
    }
}
