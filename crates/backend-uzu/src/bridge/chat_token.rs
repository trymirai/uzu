use std::{
    any::Any,
    path::PathBuf,
    pin::Pin,
    task::{Context, Poll},
};

use futures::{Stream, StreamExt, stream};
use shoji::{
    traits::{
        Instance as LlmInstance, State as LlmInstanceState,
        backend::{
            Error,
            chat_token::{StreamInput as ChatTokenStreamInput, StreamOutput as ChatTokenStreamOutput},
        },
    },
    types::session::chat::ChatReplyConfig,
};
use tokio_util::sync::CancellationToken;

use crate::{
    backends::common::Backend,
    bridge::container::Container,
    engine::{
        Engine,
        language_model::{LanguageModel, state::LanguageModelState, stream::LanguageModelIteratorData},
    },
};

pub struct UzuChatTokenLlmInstance<B: Backend> {
    model: Container<LanguageModel<B>>,
    state: Container<LanguageModelState<B>>,
}

impl<B: Backend> UzuChatTokenLlmInstance<B> {
    pub fn new(model_path: String) -> Result<Self, Error> {
        let engine = Engine::<B>::new().map_err(|err| err.to_string())?;
        let model = engine.load_language_model(&PathBuf::from(model_path)).map_err(|err| err.to_string())?;
        let state = model.create_empty_state(model.recommended_context_length()).map_err(|err| err.to_string())?;
        Ok(Self {
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
        _config: Self::StreamConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, Error>> + Send + 'a>> {
        let state = match state.as_any_mut().downcast_mut::<Container<LanguageModelState<B>>>() {
            Some(state) => state.clone(),
            None => return error_stream("unexpected state type for uzu chat token instance".to_string()),
        };
        let model = self.model.clone();

        let data = {
            let model_state = match state.value.lock() {
                Ok(model_state) => model_state,
                Err(err) => return error_stream(err.to_string()),
            };
            match LanguageModelIteratorData::new(input, &model_state) {
                Ok(data) => data,
                Err(err) => return error_stream(err.to_string()),
            }
        };

        UzuChatTokenStream {
            model,
            state,
            data,
            finished: false,
        }
        .take_until(cancel_token.cancelled_owned())
        .boxed()
    }
}

struct UzuChatTokenStream<B: Backend> {
    model: Container<LanguageModel<B>>,
    state: Container<LanguageModelState<B>>,
    data: LanguageModelIteratorData,
    finished: bool,
}

unsafe impl<B: Backend> Send for UzuChatTokenStream<B> {}

impl<B: Backend> Stream for UzuChatTokenStream<B> {
    type Item = Result<ChatTokenStreamOutput, Error>;

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        let self_mut = self.get_mut();
        let step = {
            let model_guard = match self_mut.model.value.lock() {
                Ok(model_guard) => model_guard,
                Err(err) => {
                    self_mut.finished = true;
                    return Poll::Ready(Some(Err(Box::<dyn std::error::Error + Send + Sync>::from(err.to_string()))));
                },
            };
            let mut model_state = match self_mut.state.value.lock() {
                Ok(model_state) => model_state,
                Err(err) => {
                    self_mut.finished = true;
                    return Poll::Ready(Some(Err(Box::<dyn std::error::Error + Send + Sync>::from(err.to_string()))));
                },
            };
            self_mut.data.step(&model_guard, &mut model_state)
        };

        Poll::Ready(match step.transpose() {
            Some(Ok(token)) => Some(Ok(token)),
            Some(Err(err)) => {
                self_mut.finished = true;
                Some(Err(Box::<dyn std::error::Error + Send + Sync>::from(err.to_string())))
            },
            None => {
                self_mut.finished = true;
                None
            },
        })
    }
}

impl<B: Backend> LlmInstanceState for Container<LanguageModelState<B>> {
    fn clone_boxed(&self) -> Box<dyn LlmInstanceState> {
        Box::new(self.clone())
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn error_stream<'a>(message: String) -> Pin<Box<dyn Stream<Item = Result<ChatTokenStreamOutput, Error>> + Send + 'a>> {
    Box::pin(stream::once(async move {
        Err::<ChatTokenStreamOutput, Error>(Box::<dyn std::error::Error + Send + Sync>::from(message))
    }))
}
