use std::{
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex},
};

use futures::TryStream;
use shoji::traits::{
    Backend as BackendTrait, BackendInstance as BackendInstanceTrait, LoadedModel as LoadedModelTrait,
    LoadedModelState as LoadedModelStateTrait,
    backend::message::{Input as MessageInput, Output as MessageOutput},
};

use crate::{
    TOOLCHAIN_VERSION,
    session::{ChatSession, config::DecodingConfig},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unable to load model")]
    UnableToLoad,
    #[error("Unable to create state")]
    UnableToCreateState,
    #[error("Stream failed")]
    StreamFailed,
}

#[derive(Debug, Copy, Clone)]
pub struct Backend;

impl BackendTrait for Backend {
    type Error = Error;
    type BackendInstance = BackendInstance;
    type LoadedModel = LoadedModel;
    type LoadedModelState = LoadedModelState;
    type StreamInput = MessageInput;
    type StreamOutput = MessageOutput;

    fn identifier(&self) -> String {
        "uzu".to_string()
    }

    fn version(&self) -> String {
        TOOLCHAIN_VERSION.to_string()
    }
}

pub struct BackendInstance;

impl BackendInstanceTrait for BackendInstance {
    type Backend = Backend;

    fn backend(&self) -> Self::Backend {
        Backend
    }

    fn new() -> Result<Self, <Self::Backend as BackendTrait>::Error> {
        Ok(Self)
    }

    fn load_model(
        &self,
        reference: String,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        <Self::Backend as BackendTrait>::LoadedModel,
                        <Self::Backend as BackendTrait>::Error,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            let path = PathBuf::from(&reference);
            let session = ChatSession::new(path, DecodingConfig::default()).map_err(|_| Error::UnableToLoad)?;
            Ok(LoadedModel {
                session: Arc::new(Mutex::new(session)),
            })
        })
    }
}

pub struct LoadedModel {
    session: Arc<Mutex<ChatSession>>,
}
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

impl LoadedModelTrait for LoadedModel {
    type Backend = Backend;

    fn new_state(
        &self
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        <Self::Backend as BackendTrait>::LoadedModelState,
                        <Self::Backend as BackendTrait>::Error,
                    >,
                > + Send
                + '_,
        >,
    > {
        let reset = self
            .session
            .lock()
            .map_err(|_| Error::UnableToCreateState)
            .and_then(|mut guard| guard.reset().map_err(|_| Error::UnableToCreateState));
        Box::pin(async move {
            reset?;
            Ok(LoadedModelState)
        })
    }

    fn stream(
        &self,
        _: &<Self::Backend as BackendTrait>::StreamInput,
        _: &mut <Self::Backend as BackendTrait>::LoadedModelState,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = impl TryStream<
                        Ok = <Self::Backend as BackendTrait>::StreamOutput,
                        Error = <Self::Backend as BackendTrait>::Error,
                    >,
                > + Send
                + '_,
        >,
    > {
        Box::pin(async move {
            futures::stream::iter([Err::<
                <Self::Backend as BackendTrait>::StreamOutput,
                <Self::Backend as BackendTrait>::Error,
            >(Error::StreamFailed)])
        })
    }
}

#[derive(Debug, Clone)]
pub struct LoadedModelState;

impl LoadedModelStateTrait for LoadedModelState {
    type Backend = Backend;
}
