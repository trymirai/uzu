pub mod erased;
pub mod message;
pub mod token;

use std::{error::Error, fmt::Debug, pin::Pin};

use futures::TryStream;
use serde::{Serialize, de::DeserializeOwned};

pub trait Backend: Debug + Copy + 'static {
    type Error: Error;
    type BackendInstance: BackendInstance<Backend = Self>;

    type LoadedModel: LoadedModel<Backend = Self>;
    type LoadedModelState: LoadedModelState<Backend = Self>;

    type StreamInput: Serialize + DeserializeOwned;
    type StreamOutput: Serialize + DeserializeOwned;

    fn identifier(&self) -> String;
    fn version(&self) -> String;
}

pub trait BackendInstance: Sized {
    type Backend: Backend<BackendInstance = Self>;

    fn backend(&self) -> Self::Backend;
    fn new() -> Result<Self, <Self::Backend as Backend>::Error>;

    fn load_model(
        &self,
        reference: String,
    ) -> Result<<Self::Backend as Backend>::LoadedModel, <Self::Backend as Backend>::Error>;
}

pub trait LoadedModel {
    type Backend: Backend<LoadedModel = Self>;

    fn new_state(&self) -> Result<<Self::Backend as Backend>::LoadedModelState, <Self::Backend as Backend>::Error>;

    fn stream<'a>(
        &'a self,
        input: &'a <Self::Backend as Backend>::StreamInput,
        state: &'a mut <Self::Backend as Backend>::LoadedModelState,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = impl TryStream<
                        Ok = <Self::Backend as Backend>::StreamOutput,
                        Error = <Self::Backend as Backend>::Error,
                    >,
                > + Send
                + 'a,
        >,
    >;
}

pub trait LoadedModelState: Clone {
    type Backend: Backend<LoadedModelState = Self>;
}
