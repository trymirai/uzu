pub mod chat;
pub mod reasoning_effort;
pub mod responses;

use std::{pin::Pin, sync::Arc};

use async_openai::{Client, config::OpenAIConfig};
use shoji::{
    traits::backend::{
        Error as BackendError, InstanceStream,
        chat_message::{Output, StreamMetrics},
    },
    types::session::chat::{ChatMessage, ChatReplyConfig},
};
use tokio_util::sync::CancellationToken;

pub trait ApiStream: Send + Sync {
    fn stream(
        &self,
        client: Arc<Client<OpenAIConfig>>,
        model: String,
        config: ChatReplyConfig,
        messages: Vec<ChatMessage>,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn InstanceStream<Item = Result<Output, BackendError>, Metrics = StreamMetrics> + Send>>;
}
