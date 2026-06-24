use std::pin::Pin;

use futures::{Stream, StreamExt};
use hanashi::{Encoding as EncodingTrait, chat::Encoding};
use shoji::{
    traits::{
        State,
        backend::{
            Error,
            chat_message::{Output, ToolCallState},
            chat_token::{Backend, Instance, StreamInput, StreamOutput},
        },
    },
    types::{
        basic::TokenId,
        model::Model,
        session::chat::{ChatConfig, ChatContentBlock, ChatMessage, ChatReplyConfig, ChatReplyFinishReason},
    },
};
use tokio_util::sync::CancellationToken;

use crate::chat::{
    ChatSessionError,
    helpers::{error_stream, get_encoding},
};

pub struct Session {
    instance: Box<dyn Instance>,
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
        let instance =
            backend.instance(reference.clone(), config).await.map_err(|error| ChatSessionError::Backend {
                message: error.to_string(),
            })?;
        let state = instance.state().await.map_err(|error| ChatSessionError::Backend {
            message: error.to_string(),
        })?;
        let encoding = get_encoding(reference, model).map_err(|err| ChatSessionError::Loading {
            message: err.to_string(),
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

    pub fn stream<'a>(
        &'a mut self,
        input: &'a Vec<ChatMessage>,
        config: ChatReplyConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Output, ChatSessionError>> + Send + 'a>> {
        self.input_tokens = match self.build_input(input) {
            Ok(input) => input,
            Err(err) => {
                return error_stream(ChatSessionError::Backend {
                    message: err.to_string(),
                });
            },
        };

        let encoding = &mut self.encoding;
        let stop_token_ids = &self.stop_token_ids;
        self.instance
            .stream(&self.input_tokens, self.state.as_mut(), config.clone(), cancel_token.clone())
            .map(move |event| {
                Self::build_output(
                    encoding,
                    stop_token_ids,
                    &config,
                    event,
                    cancel_token,
                    self.instance.max_context_length(),
                )
            })
            .boxed()
    }

    fn build_input(
        &mut self,
        messages: &Vec<ChatMessage>,
    ) -> Result<StreamInput, ChatSessionError> {
        let tokens_offset = self.encoding.state().tokens.len();
        let messages_offset = self.encoding.state().messages.len();
        let new_messages = messages.get(messages_offset..).ok_or_else(|| ChatSessionError::Backend {
            message: "input message history is shorter than the encoding state".to_string(),
        })?;

        self.encoding.encode(new_messages.to_vec()).map_err(|err| ChatSessionError::Backend {
            message: err.to_string(),
        })?;
        let tokens =
            self.encoding.state().tokens[tokens_offset..].iter().map(|token| token.id as u64).collect::<Vec<u64>>();
        Ok(tokens)
    }

    fn build_output(
        encoding: &mut Encoding,
        stop_token_ids: &[u64],
        config: &ChatReplyConfig,
        event: Result<StreamOutput, Error>,
        cancel_token: CancellationToken,
        max_context_length: Option<usize>,
    ) -> Result<Output, ChatSessionError> {
        if let Err(err) = event {
            return Err(ChatSessionError::Backend {
                message: err.to_string(),
            });
        }

        let token = event.unwrap();
        if let Err(err) = encoding.decode(vec![token as TokenId]) {
            return Err(ChatSessionError::Backend {
                message: err.to_string(),
            });
        }

        let message = match encoding.state().messages.last() {
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

        let tokens_count = encoding.state().tokens.len();
        let finish_reason = if cancel_token.is_cancelled() {
            Some(ChatReplyFinishReason::Cancelled)
        } else if stop_token_ids.contains(&token) {
            Some(ChatReplyFinishReason::Stop)
        } else if let Some(token_limit) = config.token_limit
            && tokens_count >= token_limit as usize
        {
            Some(ChatReplyFinishReason::Length)
        } else if let Some(length) = max_context_length
            && tokens_count >= length
        {
            Some(ChatReplyFinishReason::ContextLimitReached)
        } else {
            None
        };
        // TODO agolokoz: ask about ToolCalls and Rejected

        Ok(Output {
            reasoning: message.reasoning(),
            text: message.text(),
            tool_calls: tool_calls_states,
            finish_reason,
            // TODO agolokoz: fillstats:
            // duration: let start = Instant before next + after next,
            // tts: Instant after first token - start
            //
            stats: Default::default(),
        })
    }
}
