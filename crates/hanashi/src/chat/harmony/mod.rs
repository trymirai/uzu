mod bridging;
mod config;
mod error;

use bridging::{bridge_messages_from_harmony, bridge_messages_to_harmony};
pub use config::HarmonyConfig;
pub use error::Error;
use openai_harmony::{
    HarmonyEncoding, HarmonyEncodingName, StreamableParser,
    chat::{
        Author as HarmonyAuthor, Content as HarmonyContent, Conversation as HarmonyConversation,
        Message as HarmonyMessage, Role as HarmonyRole, TextContent as HarmonyTextContent,
    },
    load_harmony_encoding,
};
use shoji::types::{
    basic::{Token, TokenId},
    session::chat::{ChatMessage, ChatModelCapabilities},
};
use tokenizers::Tokenizer;

use crate::{
    Encoding as EncodingTrait,
    chat::{Context, State, TokenizerLocation},
};

pub struct Encoding {
    capabilities: ChatModelCapabilities,
    encoding: HarmonyEncoding,
    parser: StreamableParser,
    state: State,
    completion_message_start: usize,
}

impl EncodingTrait for Encoding {
    type Config = HarmonyConfig;
    type Context = Context;
    type Input = Vec<ChatMessage>;
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

        let encoding_name = match config {
            HarmonyConfig::GptOss => HarmonyEncodingName::HarmonyGptOss,
        };
        let encoding = load_harmony_encoding(encoding_name).map_err(|_| Error::UnableToLoadEncoding)?;
        let parser = StreamableParser::new(encoding.clone(), None).map_err(|_| Error::UnableToLoadEncoding)?;
        Ok(Self {
            capabilities: config.capabilities(),
            encoding,
            parser,
            state: State::default(),
            completion_message_start: 0,
        })
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.parser = StreamableParser::new(self.encoding.clone(), None).map_err(|_| Error::UnableToLoadEncoding)?;
        self.state = State::default();
        self.completion_message_start = 0;
        Ok(())
    }

    fn encode(
        &mut self,
        messages: Self::Input,
    ) -> Result<(), Self::Error> {
        self.state.messages.extend(messages.clone());
        self.completion_message_start = self.state.messages.len();
        // The rendered prompt ends with `<|start|>assistant`; parse only the generated continuation of that header.
        self.parser = StreamableParser::new(self.encoding.clone(), Some(HarmonyRole::Assistant))
            .map_err(|_| Error::UnableToLoadEncoding)?;

        let conversation = HarmonyConversation::from_messages(bridge_messages_to_harmony(&messages)?);
        let token_ids = self
            .encoding
            .render_conversation_for_completion(&conversation, HarmonyRole::Assistant, None)
            .map_err(|_| Error::UnableToRenderConversation)?;
        for token_id in token_ids {
            let token = resolve_token(&self.encoding, token_id)?;
            self.state.tokens.push(token);
        }

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

    fn tokenizer(&self) -> Option<&Tokenizer> {
        None
    }

    fn supports_tool_calls(&self) -> bool {
        self.capabilities.supports_tools
    }

    fn supports_multiple_tool_calls(&self) -> bool {
        self.capabilities.supports_multiple_tool_calls
    }
}

impl Encoding {
    fn update_messages_from_parser_state(&mut self) -> Result<(), Error> {
        let mut streamed_harmony_messages = self.parser.messages().to_vec();

        if let Some(current_role) = self.parser.current_role() {
            let mut content: Vec<HarmonyContent> = vec![];
            if let Ok(current_content) = self.parser.current_content() {
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
        self.state.messages.truncate(self.completion_message_start);
        self.state.messages.extend(streamed_messages);

        Ok(())
    }
}

fn resolve_token(
    encoding: &HarmonyEncoding,
    token_id: TokenId,
) -> Result<Token, Error> {
    let value = encoding.tokenizer().decode_utf8([token_id]).map_err(|_| Error::UnableToDecodeToken)?;
    let is_special = encoding.tokenizer().is_special_token(token_id);
    Ok(Token {
        id: token_id,
        value,
        is_special,
    })
}
