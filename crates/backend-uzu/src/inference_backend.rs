use std::{
    path::PathBuf,
    pin::Pin,
    sync::{Arc, Mutex},
};

use futures::{Stream, stream};
use shoji::traits::{
    Backend as BackendTrait, Instance as InstanceTrait, State as StateTrait,
    backend::{
        Error as BackendError,
        chat::StreamConfig,
        chat_token::{self, StreamInput, StreamOutput},
    },
};
use tokio_util::sync::CancellationToken;

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

pub struct Backend;

impl Backend {
    pub fn new() -> Result<Self, Error> {
        Ok(Self)
    }
}

impl BackendTrait for Backend {
    fn identifier(&self) -> String {
        "uzu".to_string()
    }

    fn version(&self) -> String {
        TOOLCHAIN_VERSION.to_string()
    }

    fn as_chat_via_token_capable(&self) -> Option<&dyn chat_token::Backend> {
        Some(self)
    }
}

impl chat_token::Backend for Backend {
    fn instance(
        &self,
        reference: String,
        _config: chat_token::Config,
    ) -> Pin<Box<dyn Future<Output = Result<Box<dyn chat_token::Instance>, BackendError>> + Send + '_>> {
        Box::pin(async move {
            let path = PathBuf::from(&reference);
            let session = ChatSession::new(path, DecodingConfig::default())
                .map_err(|_| Box::new(Error::UnableToLoad) as BackendError)?;
            Ok(Box::new(Instance {
                session: Arc::new(Mutex::new(session)),
            }) as Box<dyn chat_token::Instance>)
        })
    }
}

pub struct Instance {
    session: Arc<Mutex<ChatSession>>,
}
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

#[derive(Debug, Clone)]
pub struct State;

impl StateTrait for State {
    fn clone_boxed(&self) -> Box<dyn StateTrait> {
        Box::new(self.clone())
    }
}

impl InstanceTrait for Instance {
    type StreamConfig = StreamConfig;
    type StreamInput = StreamInput;
    type StreamOutput = StreamOutput;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn StateTrait>, BackendError>> + Send + '_>> {
        let reset = self
            .session
            .lock()
            .map_err(|_| Error::UnableToCreateState)
            .and_then(|mut guard| guard.reset().map_err(|_| Error::UnableToCreateState));
        Box::pin(async move {
            reset.map_err(|error| Box::new(error) as BackendError)?;
            Ok(Box::new(State) as Box<dyn StateTrait>)
        })
    }

    fn stream<'a>(
        &'a self,
        _input: &'a Self::StreamInput,
        _state: &'a mut dyn StateTrait,
        _config: Self::StreamConfig,
        _cancel: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, BackendError>> + Send + 'a>> {
        Box::pin(stream::once(async { Err(Box::new(Error::StreamFailed) as BackendError) }))
    }
}
