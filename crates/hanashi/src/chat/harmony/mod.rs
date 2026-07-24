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

const TOKEN_START: &str = "<|start|>";
const TOKEN_MESSAGE: &str = "<|message|>";
const TOKEN_CHANNEL: &str = "<|channel|>";
const TOKEN_CONSTRAIN: &str = "<|constrain|>";
const ROLE_ASSISTANT: &str = "assistant";
const CONTENT_TYPE_JSON: &str = "json";
const CHANNELS: [&str; 3] = ["analysis", "commentary", "final"];

pub struct Encoding {
    capabilities: ChatModelCapabilities,
    encoding: HarmonyEncoding,
    parser: StreamableParser,
    state: State,
    completion_message_start: usize,
    special_tokens: SpecialTokens,
    // Some while the current message header is being decoded (from decoding start or `<|start|>` until `<|message|>`);
    // header tokens are held back from the parser so a malformed header can be repaired before the parser sees it.
    header_buffer: Option<Vec<Token>>,
}

/// Ids of the harmony formatting tokens the decode-side header repair needs to recognize.
struct SpecialTokens {
    start: TokenId,
    message: TokenId,
    channel: TokenId,
    stop: Vec<TokenId>,
}

impl SpecialTokens {
    fn resolve(encoding: &HarmonyEncoding) -> Result<Self, Error> {
        let single = |literal: &str| -> Result<TokenId, Error> {
            match encoding.tokenizer().encode_with_special_tokens(literal).as_slice() {
                [token_id] => Ok(*token_id),
                _ => Err(Error::UnableToLoadEncoding),
            }
        };
        Ok(Self {
            start: single(TOKEN_START)?,
            message: single(TOKEN_MESSAGE)?,
            channel: single(TOKEN_CHANNEL)?,
            stop: encoding.stop_tokens().map_err(|_| Error::UnableToLoadEncoding)?.into_iter().collect(),
        })
    }
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
        let special_tokens = SpecialTokens::resolve(&encoding)?;
        Ok(Self {
            capabilities: config.capabilities(),
            encoding,
            parser,
            state: State::default(),
            completion_message_start: 0,
            special_tokens,
            header_buffer: None,
        })
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.parser = StreamableParser::new(self.encoding.clone(), None).map_err(|_| Error::UnableToLoadEncoding)?;
        self.state = State::default();
        self.completion_message_start = 0;
        self.header_buffer = None;
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
        self.header_buffer = Some(Vec::new());

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

            if self.header_buffer.is_some() {
                if token_id == self.special_tokens.message {
                    // header complete: repair it if needed, then release it to the parser
                    let header_tokens = self.header_buffer.take().unwrap_or_default();
                    let header_token_ids = repair_header(&self.encoding, &header_tokens, &self.special_tokens)
                        .unwrap_or_else(|| header_tokens.iter().map(|header_token| header_token.id).collect());
                    for header_token_id in header_token_ids {
                        self.process(header_token_id)?;
                    }
                    self.process(token_id)?;
                } else if self.special_tokens.stop.contains(&token_id) {
                    // message ended without `<|message|>`: replay verbatim and let the parser decide
                    let header_tokens = self.header_buffer.take().unwrap_or_default();
                    for header_token in header_tokens {
                        self.process(header_token.id)?;
                    }
                    self.process(token_id)?;
                } else if let Some(header_tokens) = &mut self.header_buffer {
                    header_tokens.push(token.clone());
                }
            } else {
                self.process(token_id)?;
                if token_id == self.special_tokens.start {
                    self.header_buffer = Some(Vec::new());
                }
            }

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
    fn process(
        &mut self,
        token_id: TokenId,
    ) -> Result<(), Error> {
        self.parser.process(token_id).map_err(|error| Error::ParserError {
            reason: error.to_string(),
        })?;
        Ok(())
    }

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

/// gpt-oss sometimes drops or misplaces the `<|channel|>` token when it opens a tool-call header  right after a tool response,
/// emitting e.g. `commentary to=functions.x<|channel|>commentary<|constrain|>json` straight after `<|start|>assistant`.
/// When a header is that recognizable quirk — the channel name first, followed only by fragments we can attribute
/// (a recipient, a repeated channel declaration, a content type) — rebuild the canonical header the parser expects.
/// Any other shape returns None and is replayed verbatim, keeping the parser's strict behavior for genuinely garbled output.
fn repair_header(
    encoding: &HarmonyEncoding,
    header_tokens: &[Token],
    special_tokens: &SpecialTokens,
) -> Option<Vec<TokenId>> {
    if header_tokens.first().is_some_and(|token| token.id == special_tokens.channel) {
        return None;
    }

    let header_text: String = header_tokens.iter().map(|token| token.value.as_str()).collect();
    let text = header_text.strip_prefix(ROLE_ASSISTANT).unwrap_or(&header_text).trim_start();

    let channel = CHANNELS.into_iter().find(|channel| {
        text.strip_prefix(channel).is_some_and(|rest| rest.is_empty() || rest.starts_with(' ') || rest.starts_with('<'))
    })?;

    let mut recipient: Option<&str> = None;
    let mut remainder = text[channel.len()..].trim_start();
    while !remainder.is_empty() {
        if let Some(rest) = remainder.strip_prefix("to=") {
            let end = rest.find(|c: char| c.is_whitespace() || c == '<').unwrap_or(rest.len());
            if end == 0 {
                return None;
            }
            if recipient.is_none() {
                recipient = Some(&rest[..end]);
            }
            remainder = &rest[end..];
        } else if let Some(rest) = remainder.strip_prefix(TOKEN_CHANNEL) {
            let end = rest.find(|c: char| c.is_whitespace() || c == '<').unwrap_or(rest.len());
            if &rest[..end] != channel {
                return None;
            }
            remainder = &rest[end..];
        } else if let Some(rest) = remainder.strip_prefix(TOKEN_CONSTRAIN) {
            let end = rest.find(|c: char| c.is_whitespace() || c == '<').unwrap_or(rest.len());
            remainder = &rest[end..];
        } else {
            remainder = remainder.strip_prefix(CONTENT_TYPE_JSON)?;
        }
        remainder = remainder.trim_start();
    }

    let header = match recipient {
        Some(recipient) => format!("{TOKEN_CHANNEL}{channel} to={recipient} {TOKEN_CONSTRAIN}{CONTENT_TYPE_JSON}"),
        None => format!("{TOKEN_CHANNEL}{channel}"),
    };
    Some(encoding.tokenizer().encode_with_special_tokens(&header))
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
