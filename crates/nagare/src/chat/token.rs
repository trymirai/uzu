use std::{io, pin::Pin};

use futures::{Stream, StreamExt, stream};
use hanashi::{
    Encoding as EncodingTrait,
    chat::{Config, Context, Encoding, TokenizerLocation, hanashi::Config as HanashiConfig},
};
use shoji::{
    traits::{
        State,
        backend::{
            Error as BackendError,
            chat_message::{Output, ToolCallState},
            chat_token::{Backend, Instance, StreamOutput as TokenStreamOutput},
        },
    },
    types::{
        basic::TokenId,
        model::{Model, ModelFamily},
        session::chat::{
            ChatConfig, ChatContentBlock, ChatMessage, ChatReplyConfig, ChatReplyFinishReason, ChatReplyStats,
        },
    },
};
use tokio_util::sync::CancellationToken;

use crate::chat::ChatSessionError;

pub struct Session {
    instance: Box<dyn Instance>,
    state: Box<dyn State>,
    encoding: Encoding,
    stream_tokens: Vec<u64>,
}

impl Session {
    pub async fn new(
        backend: &dyn Backend,
        config: ChatConfig,
        reference: String,
        model: &Model,
    ) -> Result<Self, ChatSessionError> {
        let instance =
            backend.instance(reference.to_string(), config).await.map_err(|error| ChatSessionError::Backend {
                message: error.to_string(),
            })?;
        let state = instance.state().await.map_err(|error| ChatSessionError::Backend {
            message: error.to_string(),
        })?;

        let encoding_config = Self::get_encoding_config(model).map_err(|err| ChatSessionError::LoadingError {
            message: err.to_string(),
        })?;
        let encoding_context = Context {
            tokenizer_location: TokenizerLocation::Directory {
                path: reference,
                name: Some("tokenizer.json".to_string()),
            },
        };
        let encoding =
            Encoding::new(encoding_config, encoding_context).map_err(|err| ChatSessionError::LoadingError {
                message: err.to_string(),
            })?;

        Ok(Self {
            instance,
            state,
            encoding,
            stream_tokens: Vec::new(),
        })
    }

    pub async fn reset(&mut self) -> Result<(), ChatSessionError> {
        self.encoding.reset().map_err(|err| ChatSessionError::Backend {
            message: err.to_string(),
        })?;
        self.state = self.instance.state().await.map_err(|err| ChatSessionError::Backend {
            message: err.to_string(),
        })?;
        Ok(())
    }

    pub fn stream<'a>(
        &'a mut self,
        input: &'a Vec<ChatMessage>,
        config: ChatReplyConfig,
        cancel_token: CancellationToken,
    ) -> Pin<Box<dyn Stream<Item = Result<Output, ChatSessionError>> + Send + 'a>> {
        self.stream_tokens = match self.get_tokens(input) {
            Ok(tok) => tok,
            Err(err) => {
                return Box::pin(stream::once(async move { Err(err) }));
            },
        };

        let token_limit = config.token_limit;
        let cancellation = cancel_token.clone();
        let backend_stream = self.instance.stream(&self.stream_tokens, self.state.as_mut(), config, cancel_token);
        let encoding = &mut self.encoding;

        let init_stream_state = TokenStreamState {
            backend_stream,
            encoding,
            output: Output::default(),
            output_token_count: 0,
            finished: false,
        };
        stream::unfold(init_stream_state, move |mut state| {
            let cancellation = cancellation.clone();
            async move {
                if state.finished {
                    return None;
                }
                if token_limit == Some(0) {
                    state.output.finish_reason = Some(ChatReplyFinishReason::Length);
                    state.finished = true;
                    return Some((Ok(state.output.clone()), state));
                }

                match state.backend_stream.next().await {
                    Some(Ok(TokenStreamOutput::Token(token))) => {
                        state.output_token_count = state.output_token_count.saturating_add(1);
                        if let Err(error) = state.encoding.decode(vec![token as TokenId]) {
                            state.finished = true;
                            return Some((
                                Err(ChatSessionError::Backend {
                                    message: error.to_string(),
                                }),
                                state,
                            ));
                        }

                        state.output = Self::build_message_output(state.encoding.state().messages.last());
                        if token_limit.is_some_and(|limit| state.output_token_count >= limit) {
                            state.output.finish_reason = Some(ChatReplyFinishReason::Length);
                            state.finished = true;
                        }
                        Some((Ok(state.output.clone()), state))
                    },
                    Some(Ok(TokenStreamOutput::Finished(finish_reason))) => {
                        state.output.finish_reason = Some(finish_reason);
                        state.finished = true;
                        Some((Ok(state.output.clone()), state))
                    },
                    Some(Err(error)) => {
                        state.finished = true;
                        Some((
                            Err(ChatSessionError::Backend {
                                message: error.to_string(),
                            }),
                            state,
                        ))
                    },
                    None => {
                        state.output.finish_reason =
                            Some(Self::stream_end_finish_reason(&cancellation, token_limit, state.output_token_count));
                        state.finished = true;
                        Some((Ok(state.output.clone()), state))
                    },
                }
            }
        })
        .boxed()
    }

    fn build_message_output(message: Option<&ChatMessage>) -> Output {
        let Some(message) = message else {
            return Output::default();
        };

        let tool_calls = message
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

        Output {
            reasoning: message.reasoning(),
            text: message.text(),
            tool_calls,
            finish_reason: None,
            stats: ChatReplyStats::default(),
        }
    }

    fn stream_end_finish_reason(
        cancellation: &CancellationToken,
        token_limit: Option<u32>,
        output_token_count: u32,
    ) -> ChatReplyFinishReason {
        if cancellation.is_cancelled() {
            ChatReplyFinishReason::Cancelled
        } else if token_limit.is_some_and(|limit| output_token_count >= limit) {
            ChatReplyFinishReason::Length
        } else {
            ChatReplyFinishReason::Stop
        }
    }

    fn get_encoding_config(model: &Model) -> Result<Config, io::Error> {
        let family: &ModelFamily =
            model.family.as_ref().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing model family"))?;
        let model_name = family.metadata.name.to_lowercase();
        let hanashi_config: HanashiConfig = serde_json::from_value(serde_json::json!({ "name": model_name }))
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err))?;
        Ok(Config::Hanashi(hanashi_config))
    }

    fn get_tokens(
        &mut self,
        messages: &Vec<ChatMessage>,
    ) -> Result<Vec<u64>, ChatSessionError> {
        let state = self.encoding.state();
        let message_offset = state.messages.len();
        let token_offset = state.tokens.len();
        let new_messages = messages.get(message_offset..).ok_or_else(|| ChatSessionError::Backend {
            message: "input message history is shorter than the encoding state".to_string(),
        })?;

        self.encoding.encode(new_messages.to_vec()).map_err(|err| ChatSessionError::Backend {
            message: err.to_string(),
        })?;
        let tokens =
            self.encoding.state().tokens[token_offset..].iter().map(|token| token.id as u64).collect::<Vec<u64>>();
        Ok(tokens)
    }
}

struct TokenStreamState<'a> {
    backend_stream: Pin<Box<dyn Stream<Item = Result<TokenStreamOutput, BackendError>> + Send + 'a>>,
    encoding: &'a mut Encoding,
    output: Output,
    output_token_count: u32,
    finished: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_end_finish_reason_reports_cancelled() {
        let cancellation = CancellationToken::new();
        cancellation.cancel();

        assert_eq!(Session::stream_end_finish_reason(&cancellation, None, 0), ChatReplyFinishReason::Cancelled);
    }

    #[test]
    fn stream_end_finish_reason_reports_length_after_token_limit() {
        assert_eq!(
            Session::stream_end_finish_reason(&CancellationToken::new(), Some(3), 3),
            ChatReplyFinishReason::Length
        );
    }

    #[test]
    fn stream_end_finish_reason_reports_stop_for_plain_stream_end() {
        assert_eq!(
            Session::stream_end_finish_reason(&CancellationToken::new(), Some(3), 2),
            ChatReplyFinishReason::Stop
        );
    }
}
