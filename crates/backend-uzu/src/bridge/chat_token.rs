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
    types::{
        basic::{SamplingMethod as ShojiSamplingMethod, SamplingPolicy as ShojiSamplingPolicy},
        session::chat::{ChatConfig, ChatReplyConfig},
    },
};
use tokio_util::sync::CancellationToken;

use crate::{
    backends::common::Backend,
    bridge::container::Container,
    encodable_block::sampling::{SamplingMethod as UzuSamplingMethod, SamplingProcessingOrder},
    engine::{
        Engine,
        language_model::{
            LanguageModel,
            state::LanguageModelState,
            stream::{LanguageModelStreamDriver, LanguageModelStreamOptions},
        },
    },
};

pub struct UzuChatTokenLlmInstance<B: Backend> {
    model: Container<LanguageModel<B>>,
    state: Container<LanguageModelState<B>>,
}

impl<B: Backend> UzuChatTokenLlmInstance<B> {
    pub fn new(
        model_path: String,
        _config: ChatConfig,
    ) -> Result<Self, Error> {
        let engine = Engine::<B>::new().map_err(|err| err.to_string())?;
        let model = engine.load_language_model(&PathBuf::from(model_path)).map_err(|err| err.to_string())?;
        let state = model.create_empty_state().map_err(|err| err.to_string())?;
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
        config: Self::StreamConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Self::StreamOutput, Error>> + Send + 'a>> {
        let state = match state.as_any_mut().downcast_mut::<Container<LanguageModelState<B>>>() {
            Some(state) => state.clone(),
            None => return error_stream("unexpected state type for uzu chat token instance".to_string()),
        };
        let model = self.model.clone();

        let driver = {
            let model = match model.value.lock() {
                Ok(model) => model,
                Err(err) => return error_stream(err.to_string()),
            };
            let model_state = match state.value.lock() {
                Ok(model_state) => model_state,
                Err(err) => return error_stream(err.to_string()),
            };
            let stream_options = get_stream_options(config, &model);
            match LanguageModelStreamDriver::new(input, &model_state, stream_options) {
                Ok(driver) => driver,
                Err(err) => return error_stream(err.to_string()),
            }
        };

        UzuChatTokenStream::new(model, state, driver).take_until(cancel_token.cancelled_owned()).boxed()
    }
}

struct UzuChatTokenStream<B: Backend> {
    model: Container<LanguageModel<B>>,
    state: Container<LanguageModelState<B>>,
    driver: LanguageModelStreamDriver,
    finished: bool,
}

impl<B: Backend> UzuChatTokenStream<B> {
    fn new(
        model: Container<LanguageModel<B>>,
        state: Container<LanguageModelState<B>>,
        driver: LanguageModelStreamDriver,
    ) -> Self {
        Self {
            model,
            state,
            driver,
            finished: false,
        }
    }
}

impl<B: Backend> Stream for UzuChatTokenStream<B> {
    type Item = Result<ChatTokenStreamOutput, Error>;

    fn poll_next(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.finished {
            return Poll::Ready(None);
        }

        let model = match this.model.value.lock() {
            Ok(model) => model,
            Err(err) => {
                this.finished = true;
                return Poll::Ready(Some(Err(Box::<dyn std::error::Error + Send + Sync>::from(err.to_string()))));
            },
        };
        let mut model_state = match this.state.value.lock() {
            Ok(model_state) => model_state,
            Err(err) => {
                this.finished = true;
                return Poll::Ready(Some(Err(Box::<dyn std::error::Error + Send + Sync>::from(err.to_string()))));
            },
        };

        Poll::Ready(
            this.driver
                .step(&model, &mut model_state)
                .map_err(|err| Box::<dyn std::error::Error + Send + Sync>::from(err.to_string()))
                .transpose(),
        )
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

fn get_stream_options<B: Backend>(
    config: ChatReplyConfig,
    model: &LanguageModel<B>,
) -> LanguageModelStreamOptions {
    let sampling_method = match config.sampling_policy {
        ShojiSamplingPolicy::Default {
            ..
        } => model.default_sampling_method(),
        ShojiSamplingPolicy::Custom {
            method,
        } => match method {
            ShojiSamplingMethod::Greedy {
                ..
            } => UzuSamplingMethod::Greedy,
            ShojiSamplingMethod::Stochastic {
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                suffix_repetition_length,
            } => UzuSamplingMethod::Stochastic {
                temperature: temperature.map(|value| value as f32),
                top_k: top_k.map(|value| value as u32),
                top_p: top_p.map(|value| value as f32),
                min_p: min_p.map(|value| value as f32),
                repetition_penalty: repetition_penalty.map(|value| value as f32),
                suffix_repetition_length: suffix_repetition_length.map(|value| value as usize),
                // TODO agolokoz: ask what to do!
                processing_order: SamplingProcessingOrder::TemperatureThenFilters,
            },
        },
    };

    LanguageModelStreamOptions {
        sampling_method,
        stop_token_ids: model.default_stop_token_ids().to_vec(),
        token_limit: config.token_limit.map(|limit| limit as usize),
    }
}
