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
    types::session::chat::{ChatConfig, ChatMessage, ChatReplyConfig},
};
use tokio_util::sync::CancellationToken;

use super::ChatSessionError;

pub struct Session {
    _instance: Box<dyn Instance>,
    _state: Box<dyn State>,
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
            _instance: instance,
            _state: state,
        })
    }

    pub async fn reset(&mut self) -> Result<(), ChatSessionError> {
        Err(ChatSessionError::Backend {
            message: "token chat session reset not implemented".to_string(),
        })
    }

    pub fn stream<'a>(
        &'a mut self,
        _input: &'a Vec<ChatMessage>,
        _config: ChatReplyConfig,
        _cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Output, ChatSessionError>> + Send + 'a>> {
        Box::pin(stream::once(async {
            Err(ChatSessionError::Backend {
                message: "Token chat session stream not implemented".to_string(),
            })
        }))
    }
}
