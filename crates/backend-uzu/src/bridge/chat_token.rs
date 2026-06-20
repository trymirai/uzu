use std::{path::PathBuf, pin::Pin};

use futures::Stream;
use shoji::{
    traits::{
        Instance as LlmInstance, State as LlmInstanceState,
        backend::{
            Error,
            chat_token::{StreamInput as ChatTokenStreamInput, StreamOutput as ChatTokenStreamOutput},
        },
    },
    types::session::chat::{ChatConfig, ChatReplyConfig},
};
use tokio_util::sync::CancellationToken;

use crate::{
    backends::common::Backend,
    bridge::container::Container,
    engine::{
        Engine,
        language_model::{LanguageModel, state::LanguageModelState},
    },
};

pub struct UzuChatTokenLlmInstance<B: Backend> {
    engine: Container<Engine<B>>,
    model: Container<LanguageModel<B>>,
    state: Container<LanguageModelState<B>>,
}

impl<B: Backend> UzuChatTokenLlmInstance<B> {
    pub fn new(
        model_path: String,
        config: ChatConfig,
    ) -> Result<Self, Error> {
        let engine = Engine::<B>::new().map_err(|err| err.to_string())?;
        let model = engine.load_language_model(&PathBuf::from(model_path)).map_err(|err| err.to_string())?;
        let state = model.create_empty_state().map_err(|err| err.to_string())?;
        Ok(Self {
            engine: Container::new(engine),
            model: Container::new(model),
            state: Container::new(state),
        })
    }
}

impl<B: Backend> LlmInstance for UzuChatTokenLlmInstance<B> {
    type StreamConfig = ChatReplyConfig;
    type StreamInput = ChatTokenStreamInput;
    type StreamOutput = ChatTokenStreamOutput;

    fn state(&self) -> Pin<Box<dyn Future<Output = Result<Box<dyn LlmInstanceState>, Error>> + Send + '_>> {
        Box::pin(async move { Ok(self.state.clone_boxed()) })
    }

    fn stream<'a>(
        &'a self,
        input: &'a Self::StreamInput,
        state: &'a mut dyn LlmInstanceState,
        config: Self::StreamConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, Error>> + Send + 'a>> {
        todo!()
    }
}

impl<B: Backend> LlmInstanceState for Container<LanguageModelState<B>> {
    fn clone_boxed(&self) -> Box<dyn LlmInstanceState> {
        Box::new(self.clone())
    }
}
