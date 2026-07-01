use std::{pin::Pin, time::Instant};

use futures::{Stream, StreamExt};
use hanashi::{Encoding as EncodingTrait, chat::Encoding};
use shoji::{
    traits::{
        State,
        backend::{
            Error,
            chat_message::{Output, ToolCallState},
            chat_token::{Backend, Instance as ChatTokenBackendInstance, StreamInput, StreamOutput},
        },
    },
    types::{
        basic::TokenId,
        model::Model,
        session::chat::{
            ChatConfig, ChatContentBlock, ChatMessage, ChatReplyConfig, ChatReplyFinishReason, ChatReplyStats,
        },
    },
};
use tokio_util::sync::CancellationToken;

use crate::chat::{
    ChatSessionError,
    helpers::{error_stream, get_encoding},
};

pub struct Session {
    instance: Box<dyn ChatTokenBackendInstance>,
    state: Box<dyn State>,
    encoding: Encoding,
    input_tokens: Vec<u64>,
    stop_token_ids: Box<[u64]>,
}

impl Session {
    pub async fn new(
        backend: &dyn Backend,
        config: ChatConfig,
        reference: String,
        model: &Model,
    ) -> Result<Self, ChatSessionError> {
        let encoding = get_encoding(reference.clone(), model).map_err(|err| ChatSessionError::Loading {
            message: err.to_string(),
        })?;
        let tokenizer = encoding.tokenizer().ok_or_else(|| ChatSessionError::Loading {
            message: "tokenizer is empty".to_string(),
        })?;

        let instance =
            backend.instance(reference, config, tokenizer).await.map_err(|error| ChatSessionError::Backend {
                message: error.to_string(),
            })?;
        let state = instance.state().await.map_err(|error| ChatSessionError::Backend {
            message: error.to_string(),
        })?;
        let stop_token_ids = instance.stop_token_ids().ok_or_else(|| ChatSessionError::Loading {
            message: "stop_token_ids is None".to_string(),
        })?;

        Ok(Self {
            instance,
            state,
            encoding,
            input_tokens: Vec::new(),
            stop_token_ids,
        })
    }

    pub async fn reset(&mut self) -> Result<(), ChatSessionError> {
        self.encoding.reset().map_err(|err| ChatSessionError::Backend {
            message: err.to_string(),
        })?;
        self.state = self.instance.state().await.map_err(|error| ChatSessionError::Backend {
            message: error.to_string(),
        })?;
        Ok(())
    }

    pub async fn stream<'a>(
        &'a mut self,
        input: &'a Vec<ChatMessage>,
        config: ChatReplyConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Output, ChatSessionError>> + Send + 'a>> {
        if let Err(err) = self.reset().await {
            return error_stream(err);
        }

        let start = Instant::now();
        self.input_tokens = match self.build_input(input) {
            Ok(input) => input,
            Err(err) => {
                return error_stream(ChatSessionError::Backend {
                    message: err.to_string(),
                });
            },
        };

        let mut state = StreamingState {
            config: config.clone(),
            cancel_token: cancel_token.clone(),
            encoding: &mut self.encoding,
            max_context_length: self.instance.max_context_length(),
            stop_token_ids: self.stop_token_ids.clone(),

            time_start: start,
            time_first_token: None,
            total_tokens_input: self.input_tokens.len(),
            total_tokens_output: 0,
        };

        self.instance
            .stream(&self.input_tokens, self.state.as_mut(), config.clone(), cancel_token)
            .map(move |event| Self::build_output(event, &mut state))
            .boxed()
    }

    pub fn peak_memory_usage(&self) -> Option<usize> {
        self.instance.peak_memory_usage()
    }

    fn build_input(
        &mut self,
        messages: &Vec<ChatMessage>,
    ) -> Result<StreamInput, ChatSessionError> {
        self.encoding.encode(messages.to_vec()).map_err(|err| ChatSessionError::Backend {
            message: err.to_string(),
        })?;
        let tokens = self.encoding.state().tokens.iter().map(|token| token.id as u64).collect::<Vec<u64>>();
        Ok(tokens)
    }

    fn build_output(
        event: Result<StreamOutput, Error>,
        state: &mut StreamingState,
    ) -> Result<Output, ChatSessionError> {
        let token = event.map_err(|err| ChatSessionError::Backend {
            message: err.to_string(),
        })?;

        let now = Instant::now();
        if state.total_tokens_output == 0 {
            state.time_first_token = Some(now)
        }
        state.total_tokens_output += 1;

        if let Err(err) = state.encoding.decode(vec![token as TokenId]) {
            return Err(ChatSessionError::Backend {
                message: err.to_string(),
            });
        }

        let message = match state.encoding.state().messages.last() {
            Some(msg) => msg,
            None => return Ok(Output::default()),
        };

        let tool_calls_states = message
            .content
            .iter()
            .filter_map(|block| match block {
                ChatContentBlock::ToolCall {
                    value,
                } => Some(ToolCallState::Finished(value.clone())),
                ChatContentBlock::ToolCallCandidate {
                    value,
                } => Some(ToolCallState::Candidate(value.json.clone())),
                _ => None,
            })
            .collect();

        let tokens_count = state.encoding.state().tokens.len();
        let finish_reason = if state.cancel_token.is_cancelled() {
            Some(ChatReplyFinishReason::Cancelled)
        } else if state.stop_token_ids.contains(&token) {
            Some(ChatReplyFinishReason::Stop)
        } else if let Some(token_limit) = state.config.token_limit
            && state.total_tokens_output >= token_limit as usize
        {
            Some(ChatReplyFinishReason::Length)
        } else if let Some(length) = state.max_context_length
            && tokens_count >= length
        {
            Some(ChatReplyFinishReason::ContextLimitReached)
        } else {
            None
        };

        Ok(Output {
            reasoning: message.reasoning(),
            text: message.text(),
            tool_calls: tool_calls_states,
            finish_reason,
            stats: state.get_stats(),
        })
    }
}

struct StreamingState<'a> {
    config: ChatReplyConfig,
    cancel_token: CancellationToken,
    encoding: &'a mut Encoding,
    max_context_length: Option<usize>,
    stop_token_ids: Box<[u64]>,

    time_start: Instant,
    time_first_token: Option<Instant>,
    total_tokens_input: usize,
    total_tokens_output: usize,
}

impl StreamingState<'_> {
    fn get_stats(&self) -> ChatReplyStats {
        let duration = Instant::now().duration_since(self.time_start).as_secs_f64();
        let time_to_first_token = self.time_first_token.map(|time| time.duration_since(self.time_start).as_secs_f64());

        let prefill_tokens_per_second = match time_to_first_token {
            Some(ttft) if ttft > 0.0 => Some(self.total_tokens_input as f64 / ttft),
            _ => None,
        };
        let generate_tokens_per_second = match time_to_first_token {
            Some(ttft) if self.total_tokens_output > 0 && duration > ttft => {
                Some((self.total_tokens_output - 1) as f64 / (duration - ttft))
            },
            _ => None,
        };

        ChatReplyStats {
            duration,
            time_to_first_token,
            prefill_tokens_per_second,
            generate_tokens_per_second,
            tokens_count_input: Some(self.total_tokens_input as u32),
            tokens_count_output: Some(self.total_tokens_output as u32),
        }
    }
}
