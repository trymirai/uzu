use std::pin::Pin;

use futures::TryStream;
use shoji::traits::{
    Backend as BackendTrait, BackendInstance as BackendInstanceTrait, LoadedModel as LoadedModelTrait,
    LoadedModelState as LoadedModelStateTrait,
    backend::message::{Input as MessageInput, Output as MessageOutput},
};

use crate::TOOLCHAIN_VERSION;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unable to load model")]
    UnableToLoad,
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
        _: String,
    ) -> Result<<Self::Backend as BackendTrait>::LoadedModel, <Self::Backend as BackendTrait>::Error> {
        Ok(LoadedModel)
    }
}

pub struct LoadedModel;

impl LoadedModelTrait for LoadedModel {
    type Backend = Backend;

    fn new_state(
        &self
    ) -> Result<<Self::Backend as BackendTrait>::LoadedModelState, <Self::Backend as BackendTrait>::Error> {
        Ok(LoadedModelState {})
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
