use std::{pin::Pin, sync::Arc};

use async_openai::{Client, config::OpenAIConfig};
use futures::{FutureExt, Stream, StreamExt, stream};
use shoji::{
    traits::backend::{Error as BackendError, chat_message::Output},
    types::session::chat::{ChatMessage, ChatReplyConfig},
};
use tokio_util::sync::CancellationToken;

use crate::openai::{
    Error,
    bridging::{
        self,
        chat::{request, stream_chunk},
    },
    stream_state::StreamState,
};

pub struct ApiStream;

impl bridging::ApiStream for ApiStream {
    fn stream(
        &self,
        client: Arc<Client<OpenAIConfig>>,
        model: String,
        config: ChatReplyConfig,
        messages: Vec<ChatMessage>,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Output, BackendError>> + Send>> {
        let stream = async move {
            let request = match request::build(&model, &config, messages) {
                Ok(request) => request,
                Err(error) => {
                    return stream::once(async move { Err(Box::new(error) as BackendError) }).boxed();
                },
            };
            match client.chat().create_stream_byot::<_, stream_chunk::Response>(request).await {
                Ok(upstream) => {
                    let mut state = StreamState::new();
                    upstream
                        .filter_map(move |response| {
                            let item = match response {
                                Ok(response) => {
                                    stream_chunk::build(response).and_then(|chunk| state.process_chunk(chunk)).map(Ok)
                                },
                                Err(error) => Some(Err(Box::new(Error::Network {
                                    message: error.to_string(),
                                }) as BackendError)),
                            };
                            futures::future::ready(item)
                        })
                        .boxed()
                },
                Err(error) => stream::once(async move {
                    Err(Box::new(Error::Network {
                        message: error.to_string(),
                    }) as BackendError)
                })
                .boxed(),
            }
        };

        stream.flatten_stream().take_until(cancel.cancelled_owned()).boxed()
    }
}
