use std::{pin::Pin, sync::Arc};

use async_openai::{Client, config::OpenAIConfig};
use futures::Stream;
use shoji::{
    traits::{
        Instance as InstanceTrait, State as StateTrait,
        backend::{
            Error as BackendError,
            chat_message::{StreamInput, StreamOutput},
        },
    },
    types::session::chat::StreamConfig,
};
use tokio_util::sync::CancellationToken;

use crate::openai::{
    ApiType,
    bridging::{self, ApiStream},
};

#[derive(Debug, Clone)]
pub struct State;

impl StateTrait for State {
    fn clone_boxed(&self) -> Box<dyn StateTrait> {
        Box::new(self.clone())
    }
}

pub struct Instance {
    client: Arc<Client<OpenAIConfig>>,
    model_identifier: String,
    api_stream: Box<dyn ApiStream>,
}

impl Instance {
    pub fn new(
        client: Arc<Client<OpenAIConfig>>,
        model_identifier: String,
        api_type: ApiType,
    ) -> Self {
        let api_stream: Box<dyn ApiStream> = match api_type {
            ApiType::Completions => Box::new(bridging::chat::api_stream::ApiStream),
            ApiType::Responses => Box::new(bridging::responses::api_stream::ApiStream),
        };
        Self {
            client,
            model_identifier,
            api_stream,
        }
    }
}

impl InstanceTrait for Instance {
    type StreamConfig = StreamConfig;
    type StreamInput = StreamInput;
    type StreamOutput = StreamOutput;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn StateTrait>, BackendError>> + Send + '_>> {
        Box::pin(async move { Ok(Box::new(State) as Box<dyn StateTrait>) })
    }

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        _state: &'a mut dyn StateTrait,
        config: Self::StreamConfig,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, BackendError>> + Send + 'a>> {
        self.api_stream.stream(self.client.clone(), self.model_identifier.clone(), config, input.clone(), cancel)
    }
}
