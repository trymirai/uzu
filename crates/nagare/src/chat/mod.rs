mod error;

pub use error::Error;
use shoji::{
    traits::backend::erased::{AnyBackend, AnyLoadedModel, AnyLoadedModelState},
    types::{Model, Specialization},
};

pub struct Session {
    loaded_model: AnyLoadedModel,
    loaded_model_state: AnyLoadedModelState,
}

impl Session {
    pub async fn new(
        backend: &AnyBackend,
        model: Model,
        path: Option<String>,
    ) -> Result<Self, Error> {
        if !model.specializations.contains(&Specialization::Chat) {
            return Err(Error::UnsupportedModel);
        }
        let reference = path.unwrap_or(model.identifier.clone());

        let loaded_model = backend.load_model(reference).await.map_err(|error| Error::Backend {
            message: error.to_string(),
        })?;
        let loaded_model_state = loaded_model.new_state().await.map_err(|error| Error::Backend {
            message: error.to_string(),
        })?;

        Ok(Self {
            loaded_model,
            loaded_model_state,
        })
    }

    pub async fn reset(&mut self) -> Result<(), Error> {
        self.loaded_model_state = self.loaded_model.new_state().await.map_err(|error| Error::Backend {
            message: error.to_string(),
        })?;
        Ok(())
    }
}
