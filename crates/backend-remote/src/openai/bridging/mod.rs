pub mod chat;
pub mod reasoning_effort;
pub mod responses;

use std::{pin::Pin, sync::Arc};

use async_openai::{Client, config::OpenAIConfig};
use futures::Stream;
use shoji::{
    traits::backend::{Error as BackendError, chat::StreamConfig, chat_message::Output},
    types::encoding::Message,
};
use tokio_util::sync::CancellationToken;

pub trait ApiStream: Send + Sync {
    fn stream(
        &self,
        client: Arc<Client<OpenAIConfig>>,
        model: String,
        config: StreamConfig,
        messages: Vec<Message>,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Output, BackendError>> + Send>>;
}
