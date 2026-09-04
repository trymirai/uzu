mod request;
mod types;

use std::{
    fs::{read_to_string, write},
    future::Future,
    path::PathBuf,
    pin::Pin,
};

use bon::bon;
use nagare::api::{Client, Error as ApiError};
pub use request::Backend;
use request::FetchModelsRequest;
use shoji::{traits::Registry as RegistryTrait, types::model::Model};
use types::Response;

use crate::{device::Device, registry::RegistryError};

pub struct Registry {
    device: Device,
    backends: Vec<Backend>,
    include_traces: bool,
    cache_path: PathBuf,
    client: Client,
}

#[bon]
impl Registry {
    #[builder]
    pub fn new(
        #[builder(into)] api_key: Option<String>,
        device: Device,
        backends: Vec<Backend>,
        #[builder(default)] include_traces: bool,
        #[builder(into)] cache_path: PathBuf,
    ) -> Result<Self, RegistryError> {
        let client =
            Client::builder().base_url("https://sdk.trymirai.com/api/v1").maybe_bearer_token(api_key).build().map_err(
                |error| RegistryError::UnableToCreate {
                    message: error.to_string(),
                },
            )?;

        Ok(Self {
            device,
            backends,
            include_traces,
            cache_path,
            client,
        })
    }
}

impl RegistryTrait for Registry {
    type Error = RegistryError;

    fn indentifier(&self) -> String {
        "mirai".to_string()
    }

    fn models(&self) -> Pin<Box<dyn Future<Output = Result<Vec<Model>, RegistryError>> + Send + '_>> {
        Box::pin(async {
            match self.fetch_models().await {
                Ok(models) => {
                    if let Err(error) = self.save_registry(&models) {
                        tracing::warn!(?error, "failed to save Mirai registry");
                    }
                    Ok(models)
                },
                Err(error) => {
                    let transient = matches!(error, ApiError::Timeout | ApiError::Network(_))
                        || matches!(error, ApiError::Http { code, .. } if code >= 500);
                    if transient && let Ok(models) = self.load_registry() {
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
        let request = FetchModelsRequest::builder()
            .device(self.device.clone())
            .backends(self.backends.clone())
            .include_traces(self.include_traces)
            .show_all(std::env::var("UZU_REGISTRY_SHOW_ALL").is_ok())
            .build();
        let response: Response = self.client.post("fetch/models", &request).await?;
        response.models().ok_or_else(|| ApiError::Decode("response contained no models".to_string()))
    }

    fn registry_path(&self) -> PathBuf {
        self.cache_path.join("registry.json")
    }

    fn save_registry(
        &self,
        models: &[Model],
    ) -> Result<(), RegistryError> {
        let contents = serde_json::to_vec_pretty(models).map_err(|error| RegistryError::UnableToGetModels {
            message: format!("Unable to serialize registry: {}", error),
        })?;
        write(self.registry_path(), contents).map_err(|error| RegistryError::UnableToGetModels {
            message: format!("Unable to write registry: {}", error),
        })
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
