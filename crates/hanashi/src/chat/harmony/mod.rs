mod bridging;
mod config;
mod encoding_name;
mod error;

use bridging::{bridge_messages_from_harmony, bridge_messages_to_harmony};
pub use config::Config;
pub use encoding_name::EncodingName;
pub use error::Error;
use openai_harmony::{
    HarmonyEncoding, StreamableParser,
    chat::{
        Author as HarmonyAuthor, Content as HarmonyContent, Conversation as HarmonyConversation,
        Message as HarmonyMessage, Role as HarmonyRole, TextContent as HarmonyTextContent,
    },
    load_harmony_encoding,
};
use shoji::types::{Message, Token, TokenId};

use crate::{
    Encoding as EncodingTrait,
    chat::{Context, State, TokenizerLocation},
};

pub struct Encoding {
    encoding: HarmonyEncoding,
    parser: StreamableParser,
    state: State,
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
        let TokenizerLocation::Directory {
            path,
            ..
        } = &context.tokenizer_location
        else {
            return Err(Error::ExpectedDirectoryTokenizerLocation);
        };
        unsafe {
            std::env::set_var("TIKTOKEN_ENCODINGS_BASE", path);
        }
        let encoding = load_harmony_encoding(config.encoding_name.into()).map_err(|_| Error::UnableToLoadEncoding)?;
        let parser = StreamableParser::new(encoding.clone(), None).map_err(|_| Error::UnableToLoadEncoding)?;
        Ok(Self {
            encoding,
            parser: parser,
            state: State::default(),
        })
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.parser = StreamableParser::new(self.encoding.clone(), None).map_err(|_| Error::UnableToLoadEncoding)?;
        self.state = State::default();
        Ok(())
    }

    fn encode(
        &mut self,
        messages: Self::Input,
    ) -> Result<(), Self::Error> {
        self.state.messages.extend(messages.clone());

        let conversation = HarmonyConversation::from_messages(bridge_messages_to_harmony(&messages)?);
        let token_ids = self
            .encoding
            .render_conversation_for_completion(&conversation, HarmonyRole::Assistant, None)
            .map_err(|_| Error::UnableToRenderConversation)?;
        for token_id in token_ids {
            let token = resolve_token(&self.encoding, token_id)?;
            self.parser.process(token_id).map_err(|error| Error::ParserError {
                reason: error.to_string(),
            })?;
            self.state.tokens.push(token);
        }

        self.update_messages_from_parser_state()?;
        Ok(())
    }

    fn decode(
        &mut self,
        token_ids: Self::Output,
    ) -> Result<(), Self::Error> {
        for token_id in token_ids {
            let token = resolve_token(&self.encoding, token_id)?;
            self.parser.process(token_id).map_err(|error| Error::ParserError {
                reason: error.to_string(),
            })?;
            self.state.tokens.push(token);
        }

        self.update_messages_from_parser_state()?;
        Ok(())
    }
}

impl Encoding {
    fn update_messages_from_parser_state(&mut self) -> Result<(), Error> {
        let mut streamed_harmony_messages = self.parser.messages().to_vec();

        if let Some(current_role) = self.parser.current_role() {
            let mut content: Vec<HarmonyContent> = vec![];
            if let Some(current_content) = self.parser.current_content().ok() {
                content.push(HarmonyContent::Text(HarmonyTextContent {
                    text: current_content,
                }));
            }
            streamed_harmony_messages.push(HarmonyMessage {
                author: HarmonyAuthor::from(current_role),
                channel: self.parser.current_channel(),
                content,
                content_type: self.parser.current_content_type(),
                recipient: self.parser.current_recipient(),
            });
        }

        let streamed_messages = bridge_messages_from_harmony(&streamed_harmony_messages)?;
        self.state.synchronize_messages(&streamed_messages)?;

        Ok(())
    }
}

fn resolve_token(
    encoding: &HarmonyEncoding,
    token_id: TokenId,
) -> Result<Token, Error> {
    let value = encoding.tokenizer().decode_utf8(&[token_id]).map_err(|_| Error::UnableToDecodeToken)?;
    let is_special = encoding.tokenizer().is_special_token(token_id);
    Ok(Token {
        id: token_id,
        value,
        is_special,
    })
}
