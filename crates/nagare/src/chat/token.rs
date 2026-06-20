use std::{io, pin::Pin};

use futures::{Stream, StreamExt, stream};
use hanashi::{
    Encoding as EncodingTrait,
    chat::{Config, Context, Encoding, Error, TokenizerLocation, hanashi::Config as HanashiConfig},
};
use shoji::{
    traits::{
        State,
        backend::{
            chat_message::{Output, ToolCallState},
            chat_token::{Backend, Instance},
        },
    },
    types::{
        basic::TokenId,
        model::{Model, ModelFamily},
        session::chat::{ChatConfig, ChatContentBlock, ChatMessage, ChatReplyConfig, ChatReplyStats},
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
        self.stream_tokens = match self.get_tokens(input) {
            Ok(tok) => tok,
            Err(err) => {
                return Box::pin(stream::once(async move {
                    Err(ChatSessionError::Backend {
                        message: err.to_string(),
                    })
                }));
            },
        };

        let encoding = &mut self.encoding;
        self.instance
            .stream(&self.stream_tokens, self.state.as_mut(), config, cancel_token)
            .map(move |event| {
                let token = event.map_err(|error| ChatSessionError::Backend {
                    message: error.to_string(),
                })?;
                encoding.decode(vec![token as TokenId]).map_err(|error| ChatSessionError::Backend {
                    message: error.to_string(),
                })?;
                Ok(Self::build_output(encoding.state().messages.last()))
            })
            .boxed()
    }

    fn build_output(message: Option<&ChatMessage>) -> Output {
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
    ) -> Result<Vec<u64>, Error> {
        self.encoding.reset()?;
        self.encoding.encode(messages.to_vec())?;
        let tokens = self.encoding.state().tokens.iter().map(|token| token.id as u64).collect::<Vec<u64>>();
        Ok(tokens)
    }
}
