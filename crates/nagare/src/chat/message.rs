use std::pin::Pin;

use futures::{Stream, StreamExt};
use shoji::{
    traits::{
        State,
        backend::chat_message::{Backend, Instance, Output},
    },
    types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
};
use tokio_util::sync::CancellationToken;

use super::ChatSessionError;

pub struct Session {
    instance: Box<dyn Instance>,
    state: Box<dyn State>,
}

impl Session {
    pub async fn new(
        backend: &dyn Backend,
        config: ChatConfig,
        reference: String,
    ) -> Result<Self, ChatSessionError> {
        let instance = backend.instance(reference, config).await.map_err(|error| ChatSessionError::Backend {
            message: error.to_string(),
        })?;
        let state = instance.state().await.map_err(|error| ChatSessionError::Backend {
            message: error.to_string(),
        })?;
        Ok(Self {
            instance,
            state,
        })
    }

    pub async fn reset(&mut self) -> Result<(), ChatSessionError> {
        self.state = self.instance.state().await.map_err(|error| ChatSessionError::Backend {
            message: error.to_string(),
        })?;
        Ok(())
    }

    pub fn stream<'a>(
        &'a mut self,
        input: &'a Vec<ChatMessage>,
        config: ChatReplyConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Output, ChatSessionError>> + Send + 'a>> {
        self.instance
            .stream(input, self.state.as_mut(), config, cancel_token)
            .map(|event| {
                event.map_err(|error| ChatSessionError::Backend {
                    message: error.to_string(),
                })
            })
            .boxed()
    }
}
