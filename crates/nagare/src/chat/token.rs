use std::pin::Pin;

use futures::{Stream, stream};
use shoji::{
    traits::{
        State,
        backend::{
            chat_message::Output,
            chat_token::{Backend, Instance},
        },
    },
    types::{
        encoding::Message,
        session::chat::{Config, StreamConfig},
    },
};
use tokio_util::sync::CancellationToken;

use super::Error;

pub struct Session {
    _instance: Box<dyn Instance>,
    _state: Box<dyn State>,
}

impl Session {
    pub async fn new(
        backend: &dyn Backend,
        config: Config,
        reference: String,
    ) -> Result<Self, Error> {
        let instance = backend.instance(reference, config).await.map_err(|error| Error::Backend {
            message: error.to_string(),
        })?;
        let state = instance.state().await.map_err(|error| Error::Backend {
            message: error.to_string(),
        })?;
        Ok(Self {
            _instance: instance,
            _state: state,
        })
    }

    pub async fn reset(&mut self) -> Result<(), Error> {
        Err(Error::Backend {
            message: "token chat session reset not implemented".to_string(),
        })
    }

    pub fn stream<'a>(
        &'a mut self,
        _input: &'a Vec<Message>,
        _config: StreamConfig,
        _cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Output, Error>> + Send + 'a>> {
        Box::pin(stream::once(async {
            Err(Error::Backend {
                message: "Token chat session stream not implemented".to_string(),
            })
        }))
    }
}
