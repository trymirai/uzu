pub mod config;
mod error;
pub mod messages;
mod ordering;
pub mod renderer;
mod token;

use std::path::PathBuf;

pub use config::Config;
pub use error::Error;
use shoji::types::{
    basic::{Token, TokenId},
    session::chat::{ContentBlock, Message},
};
use token_stream_parser::{Parser as _, token_stream::TokenStreamParser};
use tokenizers::{Tokenizer, step_decode_stream};

use self::{config::ResolvedConfig, messages::streamed::Message as StreamedMessage, ordering::Validator};
use crate::{
    Encoding as EncodingTrait,
    chat::{
        Context, State, SynchronizationError, SynchronizationResult, TokenizerLocation,
        hanashi::{messages::rendered::FieldConfig, renderer::Renderer, token::ToParserToken},
    },
};

const DEFAULT_TOKENIZER_FILENAME: &str = "tokenizer.json";

pub struct Encoding {
    config: ResolvedConfig,
    tokenizer: Tokenizer,
    parser: TokenStreamParser,
    renderer: Renderer,
    validator: Validator,

    state: State,
    tokenizer_decode_ids: Vec<u32>,
    tokenizer_decode_prefix: String,
    tokenizer_decode_prefix_index: usize,
}

impl EncodingTrait for Encoding {
    type Config = Config;
    type Context = Context;
    type Input = Vec<Message>;
    type Output = Vec<TokenId>;
    type State = State;
    type Error = Error;

    fn new(
        config: Self::Config,
        context: Self::Context,
    ) -> Result<Self, Self::Error> {
        let config = config.resolve()?;
        let tokenizer_path = match &context.tokenizer_location {
            TokenizerLocation::File {
                path,
            } => PathBuf::from(path),
            TokenizerLocation::Directory {
                path,
                name,
            } => match name {
                Some(name) => PathBuf::from(path).join(name),
                None => PathBuf::from(path).join(DEFAULT_TOKENIZER_FILENAME),
            },
        };
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|_| Error::UnableToLoadTokenizer)?;
        let parser = TokenStreamParser::new(config.parsing.clone())?;
        let renderer = Renderer::new(config.rendering.clone());
        let validator = Validator::new(config.ordering.clone());
        Ok(Self {
            config,
            tokenizer,
            parser,
            renderer,
            validator,
            state: State::default(),
            tokenizer_decode_ids: vec![],
            tokenizer_decode_prefix: "".to_string(),
            tokenizer_decode_prefix_index: 0,
        })
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.parser.reset();
        self.validator.reset();
        self.state = State::default();
        self.tokenizer_decode_ids = vec![];
        self.tokenizer_decode_prefix = "".to_string();
        self.tokenizer_decode_prefix_index = 0;
        Ok(())
    }

    fn encode(
        &mut self,
        messages: Self::Input,
    ) -> Result<(), Self::Error> {
        let messages = self.fill_default_content(&messages)?;
        for message in &messages {
            self.validator.validate_next(&message.role)?;
        }
        self.state.messages.extend(messages.clone());

        let bos_token = self.config.tokens.bos_token_id.and_then(|token_id| self.resolve_token(token_id, false).ok());
        let eos_token = self.config.tokens.eos_token_id.and_then(|token_id| self.resolve_token(token_id, false).ok());
        let text = self.renderer.render(&messages, true, bos_token, eos_token, None)?;
        let text_encoding = self.tokenizer.encode(text, false).map_err(|_| Error::UnableToEncodeText)?;
        for token_id in text_encoding.get_ids() {
            let token = self.resolve_token(*token_id, true)?;
            self.parser.push(&token.clone().to_parser_token())?;
            self.state.tokens.push(token);
        }

        self.update_messages_from_parser_state()?;
        Ok(())
    }

    fn decode(
        &mut self,
        token_ids: Self::Output,
    ) -> Result<(), Self::Error> {
        for token_id in &token_ids {
            let token = self.resolve_token(*token_id, true)?;
            self.parser.push(&token.clone().to_parser_token())?;
            self.state.tokens.push(token);
        }

        self.update_messages_from_parser_state()?;
        Ok(())
    }
}

impl Encoding {
    fn fill_default_content(
        &self,
        messages: &[Message],
    ) -> Result<Vec<Message>, Error> {
        let mut modified_messages = Vec::new();
        for message in messages {
            let mut modified_message = message.clone();
            if let Some(role_config) = self.config.rendering.rendering.get(&modified_message.role) {
                for (_, field) in role_config.message.iter().chain(role_config.context.iter()) {
                    if !field.required {
                        continue;
                    }

                    match &field.config {
                        FieldConfig::Unique {
                            block,
                            allowed_values: Some(allowed_values),
                            ..
                        } => {
                            if !allowed_values.len() == 1 {
                                continue;
                            }
                            if let Some(expected_value) =
                                allowed_values.first().cloned().and_then(|value| value.as_str().map(|s| s.to_string()))
                            {
                                if !modified_message
                                    .content
                                    .iter()
                                    .any(|message_block| message_block.get_type() == *block)
                                {
                                    modified_message.content.insert(
                                        0,
                                        ContentBlock::Text {
                                            value: expected_value,
                                        },
                                    );
                                }
                            }
                        },
                        _ => {},
                    }
                }
            }
            modified_messages.push(modified_message);
        }
        Ok(modified_messages)
    }

    fn update_messages_from_parser_state(&mut self) -> Result<(), Error> {
        let value = self.parser.state().value.clone();
        let rendering_config = &self.config.rendering;
        let streamed_messages: Vec<Message> = serde_json::from_value::<Vec<StreamedMessage>>(value)
            .map_err(|_| Error::InvalidStreamedContent)?
            .into_iter()
            .map(|streamed_message| {
                let role = rendering_config.get_role_by_name(&streamed_message.role.to_string());
                let mut message = Message::from(streamed_message);
                message.role = role;
                message
            })
            .collect();

        let result = self.state.synchronize_messages(&streamed_messages)?;
        match result {
            SynchronizationResult::Inserted => {
                let last_message = self.state.messages.last().ok_or(SynchronizationError::Desynchronization)?;
                self.validator.validate_next(&last_message.role)?;
            },
            _ => {},
        }

        Ok(())
    }

    fn resolve_token(
        &mut self,
        token_id: TokenId,
        from_stream: bool,
    ) -> Result<Token, Error> {
        let value = if from_stream {
            step_decode_stream(
                &self.tokenizer,
                vec![token_id],
                false,
                &mut self.tokenizer_decode_ids,
                &mut self.tokenizer_decode_prefix,
                &mut self.tokenizer_decode_prefix_index,
            )
            .map_err(|_| Error::UnableToDecodeToken)?
            .unwrap_or("".to_string())
        } else {
            self.tokenizer.decode(&[token_id], false).map_err(|_| Error::UnableToDecodeToken)?
        };
        let is_special = self.tokenizer.get_added_vocabulary().is_special_token(&value);
        Ok(Token {
            id: token_id,
            value,
            is_special,
        })
    }
}
