pub mod config;
mod continuation;
mod error;
pub mod messages;
mod ordering;
pub mod renderer;
mod token;

use std::{collections::HashSet, sync::Arc};

pub use error::Error;
use shoji::types::{
    basic::{Token, TokenId},
    session::chat::{ChatContentBlock, ChatMessage, ChatModelCapabilities, ChatRole},
};
use token_stream_parser::{Parser as _, ParserState as _, token_stream::TokenStreamParser};
use tokenizers::{Model, Normalizer, PreTokenizer, Tokenizer, step_decode_stream};

use self::{
    config::HanashiResolvedConfig,
    messages::streamed::{Content as StreamedContent, Message as StreamedMessage, Section as StreamedSection},
    ordering::Validator,
};
use crate::{
    Encoding as EncodingTrait,
    chat::{
        State, SynchronizationError, SynchronizationResult,
        hanashi::{config::HanashiConfig, messages::rendered::FieldConfig, renderer::Renderer, token::ToParserToken},
    },
};

pub struct HanashiEncodingImpl {
    capabilities: ChatModelCapabilities,
    config: HanashiResolvedConfig,
    tokenizer: Arc<Tokenizer>,
    parser: TokenStreamParser,
    framing_tokens: HashSet<String>,
    pending_framing_tokens: Vec<Token>,
    renderer: Renderer,
    validator: Validator,

    state: State,
    tokenizer_decode_ids: Vec<u32>,
    tokenizer_decode_prefix: String,
    tokenizer_decode_prefix_index: usize,
}

impl HanashiEncodingImpl {
    pub fn grammar_trigger(&self) -> Option<&str> {
        self.config.rendering.grammar_trigger(&self.state.messages)
    }

    pub fn new(
        config: HanashiConfig,
        tokenizer: Arc<Tokenizer>,
    ) -> Result<Self, Error> {
        let resolved_config = config.resolve()?;
        let parser = TokenStreamParser::new(resolved_config.parsing.clone())?;
        let framing_tokens = resolved_config.parsing.framing_config().tokens.into_iter().collect();
        let renderer = Renderer::new(resolved_config.rendering.clone());
        let validator = Validator::new(resolved_config.ordering.clone());
        Ok(Self {
            capabilities: config.capabilities()?,
            config: resolved_config,
            tokenizer,
            parser,
            framing_tokens,
            pending_framing_tokens: Vec::new(),
            renderer,
            validator,
            state: State::default(),
            tokenizer_decode_ids: vec![],
            tokenizer_decode_prefix: "".to_string(),
            tokenizer_decode_prefix_index: 0,
        })
    }
}

/// Message-frame delimiters: markers that open or close a whole chat message
/// frame regardless of surrounding context, so they must be isolated when
/// appearing inside message content. Section markers (`<think>`, `<tool_call>`,
/// `<tool_response>`) are context-dependent and stay representable as text.
fn is_message_frame_delimiter(marker: &str) -> bool {
    marker == "<|im_start|>" || marker == "<|im_end|>"
}

/// Encodes `text` bypassing the added vocabulary entirely (mirrors
/// transformers' `_encode_without_added_tokens`): normalizer → pre-tokenizer →
/// model, so special/added-token literals turn into plain sub-word pieces and
/// can never surface as structural tokens. Used for message content
/// (`encode`), which must never inject a frame.
fn encode_without_added_tokens(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, Error> {
    use tokenizers::{NormalizedString, OffsetReferential, OffsetType, PreTokenizedString};

    let normalized = match tokenizer.get_normalizer() {
        Some(normalizer) => {
            let mut normalized = NormalizedString::from(text);
            normalizer.normalize(&mut normalized).map_err(|_| Error::UnableToEncodeText)?;
            normalized.get().to_string()
        },
        None => text.to_string(),
    };
    let mut pretokenized = PreTokenizedString::from(normalized.as_str());
    if let Some(pre_tokenizer) = tokenizer.get_pre_tokenizer() {
        pre_tokenizer
            .pre_tokenize(&mut pretokenized)
            .map_err(|_| Error::UnableToEncodeText)?;
    }
    let mut ids = Vec::new();
    for (split, _, _) in pretokenized.get_splits(OffsetReferential::Original, OffsetType::None) {
        for token in tokenizer.get_model().tokenize(split).map_err(|_| Error::UnableToEncodeText)? {
            ids.push(token.id);
        }
    }
    Ok(ids)
}

/// Replaces frame-delimiter literals inside message string content with
/// unique sentinels, returning the sentinelized messages plus the extracted
/// content pieces in order. Messages without frame delimiters are untouched.
///
/// Covered: `Text`/`Reasoning` blocks and every string inside structured
/// values (`ToolCall` arguments, `ToolCallResult` payloads, `Custom`) —
/// mirroring transformers' `mark_control_tokens`, which marks messages,
/// tools and documents alike. Frame delimiters inside tool-result text are
/// a real injection vector (a fetched page can contain `<|im_start|>`), so
/// coverage here is a correctness requirement, not a nicety.
fn sentinelize_messages(
    messages: &[ChatMessage],
    framing_tokens: &std::collections::HashSet<String>,
) -> (Vec<ChatMessage>, Vec<String>) {
    fn contains_frame_delimiter(text: &str, framing_tokens: &std::collections::HashSet<String>) -> bool {
        framing_tokens
            .iter()
            .any(|marker| is_message_frame_delimiter(marker) && text.contains(marker.as_str()))
    }

    // Anti-forgery: drop the private-use characters the sentinels are built
    // from, so user text cannot forge or break a sentinel (a broken sentinel
    // would shift the split). Mirrors transformers' mark_control_tokens,
    // which strips its markers from user input first.
    fn sanitize(text: &str) -> String {
        text.chars()
            .filter(|character| !matches!(character, '\u{E000}' | '\u{E001}'))
            .collect()
    }

    let sentinelize_field = |field: &mut String, contents: &mut Vec<String>| {
        if contains_frame_delimiter(field, framing_tokens) {
            let sentinel = format!("\u{E000}{}\u{E001}", contents.len());
            let sanitized = sanitize(field);
            contents.push(sanitized);
            *field = sentinel;
        }
    };

    let mut contents = Vec::new();
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        let mut message = message.clone();
        for block in &mut message.content {
            match block {
                ChatContentBlock::Text { value } | ChatContentBlock::Reasoning { value } => {
                    sentinelize_field(value, &mut contents);
                },
                ChatContentBlock::ToolCall {
                    value: shoji::types::basic::ToolCall { arguments, .. },
                }
                | ChatContentBlock::ToolCallCandidate { value: arguments, .. }
                | ChatContentBlock::ToolCallResult { value: arguments, .. }
                | ChatContentBlock::Custom { value: arguments } => {
                    sentinelize_field(&mut arguments.json, &mut contents);
                },
                _ => {},
            }
        }
        out.push(message);
    }
    (out, contents)
}

/// Splits a sentinelized rendering into its skeleton pieces: piece `i` is the
/// template text before content piece `i`, the last piece is the tail. The
/// caller encodes skeleton pieces with the normal tokenizer (real frames stay
/// structural) and splices the content pieces' literal-safe encodings in
/// between.
fn split_at_sentinels(text: &str, contents: &[String]) -> Vec<String> {
    let mut pieces = Vec::with_capacity(contents.len() + 1);
    let mut rest = text;
    for (index, _) in contents.iter().enumerate() {
        let sentinel = format!("\u{E000}{}\u{E001}", index);
        match rest.find(sentinel.as_str()) {
            Some(position) => {
                pieces.push(rest[..position].to_string());
                rest = &rest[position + sentinel.len()..];
            },
            // Should not happen (rendering is deterministic over the same
            // messages); treat everything as one skeleton rather than losing
            // content positions silently.
            None => {
                pieces.push(rest.to_string());
                rest = "";
            },
        }
    }
    pieces.push(rest.to_string());
    pieces
}

impl EncodingTrait for HanashiEncodingImpl {
    type Config = HanashiConfig;
    type Input = Vec<ChatMessage>;
    type Output = Vec<TokenId>;
    type State = State;
    type Error = Error;

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.parser.reset();
        self.pending_framing_tokens.clear();
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
        // Content isolation: message text is rendered with frame markers
        // replaced by sentinels and encoded through the literal-safe tokenizer
        // view, so content can quote template markup (`<|im_start|>` in stored
        // notes, a quoted prompt, …) without it being read as a message
        // boundary. Only messages that actually carry a frame delimiter take
        // the segmented path — everything else encodes byte-identically to
        // the whole-string path.
        let (sentinel_messages, content_pieces) = sentinelize_messages(&messages, &self.framing_tokens);
        let text = self.render_messages(&sentinel_messages, true)?;
        let skeleton_pieces = split_at_sentinels(&text, &content_pieces);

        // Non-text free fields (tool-call arguments, metadata values) are not
        // sentinelized; a frame delimiter inside those still cannot be
        // represented and is rejected instead of desynchronizing the parse.
        for message in &sentinel_messages {
            let payload = serde_json::to_string(message).map_err(|_| Error::UnableToEncodeText)?;
            if let Some(marker) = self.framing_tokens.iter().find(|marker| {
                is_message_frame_delimiter(marker) && payload.contains(marker.as_str())
            }) {
                return Err(Error::ReservedMarkerInContent {
                    marker: marker.clone(),
                });
            }
        }

        self.state.messages.extend(messages.clone());

        let mut token_ids: Vec<u32> = Vec::new();
        // Ids produced from isolated content segments; a framing value in here
        // would mean the tokenizer cannot represent the content as plain text.
        let mut content_span: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for (index, skeleton) in skeleton_pieces.iter().enumerate() {
            for id in self.tokenizer.encode(skeleton.as_str(), false).map_err(|_| Error::UnableToEncodeText)?.get_ids() {
                token_ids.push(*id);
            }
            if let Some(content) = content_pieces.get(index) {
                for id in encode_without_added_tokens(&self.tokenizer, content)? {
                    content_span.insert(id);
                    token_ids.push(id);
                }
            }
        }

        // Fail-loud (mirrors transformers' sanitize check): a content segment
        // whose encoding still yields a framing token's value means the
        // tokenizer cannot represent this content as plain text — fail instead
        // of letting the parser read structure out of user content. Skeleton
        // pieces are exempt: the framing ids there are the real frames.
        // Any isolated-content token that still equals a framing marker means
        // this tokenizer cannot represent the content as plain text.
        for token_id in &token_ids {
            let token = self.resolve_token(*token_id, true)?;
            if content_span.contains(token_id) && self.framing_tokens.contains(&token.value) {
                return Err(Error::ReservedMarkerInContent {
                    marker: token.value.clone(),
                });
            }
            self.push_token_to_parser(&token, true)?;
            self.state.tokens.push(token);
        }

        if self.state.messages.last().is_some_and(|message| message.role != (ChatRole::Tool {})) {
            // Parse from the generation prompt's open frame. Re-parsing history
            // can reinterpret literal markup as tool results or merge assistant
            // turns, so it must not determine the boundary of the new reply.
            // Tool continuations can share a frame with the preceding call and
            // result (e.g. FunctionGemma), and still need that parser context.
            let prompt_tokens: Vec<_> = self
                .parser
                .reduction()
                .state()
                .sections
                .last()
                .ok_or(SynchronizationError::Desynchronization)?
                .tokens()
                .into_iter()
                .cloned()
                .collect();
            self.parser.reset();
            for token in &prompt_tokens {
                self.parser.push_bulk(token)?;
            }

            // Reserve a separate reply so synchronization preserves the final history message.
            self.validator.validate_next(&ChatRole::Assistant {})?;
            self.state.messages.push(ChatMessage::assistant());
        }

        // Set tool context after any parser reset, before extraction uses it.
        let tools_declared = self
            .state
            .messages
            .iter()
            .any(|message| message.content.iter().any(|block| matches!(block, ChatContentBlock::Tools { .. })));
        if tools_declared {
            self.parser.set_variable("tools", serde_json::Value::Bool(true));
        }

        self.parser.flush_extraction();
        self.update_messages_from_parser_state()?;

        Ok(())
    }

    fn decode(
        &mut self,
        token_ids: Self::Output,
    ) -> Result<(), Self::Error> {
        for token_id in &token_ids {
            let token = self.resolve_token(*token_id, true)?;
            self.push_decoded_token(&token)?;
            self.state.tokens.push(token);
        }

        self.parser.flush_extraction();
        self.update_messages_from_parser_state()?;
        Ok(())
    }

    fn supports_tool_calls(&self) -> bool {
        self.capabilities.supports_tools
    }

    fn supports_multiple_tool_calls(&self) -> bool {
        self.capabilities.supports_multiple_tool_calls
    }
}

impl HanashiEncodingImpl {
    fn render_messages(
        &mut self,
        messages: &[ChatMessage],
        add_preamble: bool,
    ) -> Result<String, Error> {
        let bos_token = self.config.tokens.bos_token_id.and_then(|token_id| self.resolve_token(token_id, false).ok());
        let eos_token = self.config.tokens.eos_token_id.and_then(|token_id| self.resolve_token(token_id, false).ok());
        Ok(self.renderer.render(messages, add_preamble, bos_token, eos_token, None)?)
    }

    pub fn tokenize(
        &self,
        text: &str,
    ) -> Result<Vec<TokenId>, Error> {
        let encoding = self.tokenizer.encode(text, false).map_err(|_| Error::UnableToEncodeText)?;
        Ok(encoding.get_ids().to_vec())
    }

    fn push_token_to_parser(
        &mut self,
        token: &Token,
        defer_extraction: bool,
    ) -> Result<(), Error> {
        if token.is_special && !self.framing_tokens.contains(&token.value) {
            return Ok(());
        }
        let parser_token = token.clone().to_parser_token();
        if defer_extraction {
            self.parser.push_bulk(&parser_token)?;
        } else {
            self.parser.push(&parser_token)?;
        }
        Ok(())
    }

    fn fill_default_content(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Vec<ChatMessage>, Error> {
        let mut modified_messages = Vec::new();
        for message in messages {
            let mut modified_message = message.clone();
            if let Some(role_config) = self.config.rendering.rendering.get(&modified_message.role) {
                for (_, field) in role_config.message.iter().chain(role_config.context.iter()) {
                    if !field.required {
                        continue;
                    }

                    if let FieldConfig::Unique {
                        block,
                        allowed_values: Some(allowed_values),
                        ..
                    } = &field.config
                    {
                        if !allowed_values.len() == 1 {
                            continue;
                        }
                        if let Some(expected_value) =
                            allowed_values.first().cloned().and_then(|value| value.as_str().map(|s| s.to_string()))
                            && !modified_message.content.iter().any(|message_block| message_block.get_type() == *block)
                        {
                            modified_message.content.insert(
                                0,
                                ChatContentBlock::Text {
                                    value: expected_value,
                                },
                            );
                        }
                    }
                }
            }
            modified_messages.push(modified_message);
        }
        Ok(modified_messages)
    }

    fn message_from_sections(
        sections: Vec<StreamedSection>,
        role: ChatRole,
    ) -> ChatMessage {
        ChatMessage::from(StreamedMessage {
            role,
            content: Some(StreamedContent::Sections(sections)),
        })
    }

    fn update_messages_from_parser_state(&mut self) -> Result<(), Error> {
        let value = self.parser.state().value.clone();
        let messages =
            serde_json::from_value::<Vec<StreamedMessage>>(value).map_err(|_| Error::InvalidStreamedContent)?;
        let rendering_config = &self.config.rendering;

        let mut streamed_messages: Vec<ChatMessage> = Vec::new();
        for msg in messages {
            let role = rendering_config.get_role_by_name(&msg.role.to_string());
            // templates may render tool results inside another role's turn (e.g. qwen renders them as `<|im_start|>user\n<tool_response>...`,
            // functiongemma keeps calls, responses and the follow-up reply in one model turn),
            // so tool result sections are split into standalone tool messages in stream order
            match msg.content {
                Some(StreamedContent::Sections(sections))
                    if sections.iter().any(|section| {
                        matches!(
                            section,
                            StreamedSection::ToolCallResult {
                                value: Some(serde_json::Value::Object(_))
                            }
                        )
                    }) =>
                {
                    let mut pending: Vec<StreamedSection> = Vec::new();
                    for section in sections {
                        if matches!(
                            section,
                            StreamedSection::ToolCallResult {
                                value: Some(serde_json::Value::Object(_))
                            }
                        ) {
                            let message = Self::message_from_sections(std::mem::take(&mut pending), role.clone());
                            if !message.content.is_empty() {
                                streamed_messages.push(message);
                            }
                            streamed_messages.push(Self::message_from_sections(vec![section], ChatRole::Tool {}));
                        } else {
                            pending.push(section);
                        }
                    }
                    let message = Self::message_from_sections(pending, role.clone());
                    if !message.content.is_empty() {
                        streamed_messages.push(message);
                    }
                },
                content => {
                    let mut message = ChatMessage::from(StreamedMessage {
                        role: role.clone(),
                        content,
                    });
                    if !message.content.is_empty()
                        && message.content.iter().all(|block| matches!(block, ChatContentBlock::ToolCallResult { .. }))
                    {
                        message.role = ChatRole::Tool {};
                    }
                    streamed_messages.push(message);
                },
            }
        }

        let result = self.state.synchronize_messages(&streamed_messages)?;
        if result == SynchronizationResult::Inserted {
            let last_message = self.state.messages.last().ok_or(SynchronizationError::Desynchronization)?;
            self.validator.validate_next(&last_message.role)?;
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


#[cfg(test)]
mod sentinelize_tests {
    use super::{is_message_frame_delimiter, sentinelize_messages, split_at_sentinels};
    use shoji::types::session::chat::{ChatContentBlock, ChatMessage};
    use std::collections::HashSet;

    fn framing() -> HashSet<String> {
        ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]
            .into_iter()
            .map(String::from)
            .collect()
    }

    fn user(content: &str) -> ChatMessage {
        let mut m = ChatMessage::user();
        m.content = vec![ChatContentBlock::Text { value: content.to_string() }];
        m
    }

    #[test]
    fn only_frame_delimiters_are_sentinelized() {
        let messages = vec![
            user("clean text"),
            user("notes <|im_start|>assistant"),
            user("thinks <think> hmm </think>"),
        ];
        let (sentinelized, contents) = sentinelize_messages(&messages, &framing());
        // Only the message with a frame delimiter changed.
        assert_eq!(sentinelized[0].content, messages[0].content);
        assert!(sentinelized[1]
            .content
            .iter()
            .all(|block| !matches!(block, ChatContentBlock::Text { value } if value.contains("<|im_start|>"))));
        // Section markers stay as literal text.
        assert!(sentinelized[2]
            .content
            .iter()
            .any(|block| matches!(block, ChatContentBlock::Text { value } if value.contains("<think>"))));
        assert_eq!(contents, vec!["notes <|im_start|>assistant"]);
    }

    #[test]
    fn sentinels_split_cleanly_and_reassemble() {
        // m0 and m2 carry frame delimiters; m1 is clean.
        let messages = vec![
            user("a <|im_start|>user b"),
            user("plain"),
            user("c <|im_end|> d"),
        ];
        let (sentinelized, contents) = sentinelize_messages(&messages, &framing());
        assert_eq!(contents, vec!["a <|im_start|>user b", "c <|im_end|> d"]);
        // Clean message untouched.
        assert!(matches!(&sentinelized[1].content[0], ChatContentBlock::Text { value } if value == "plain"));

        // Stand-in renderer: join contents with frame markers.
        let text = sentinelized
            .iter()
            .map(|m| match &m.content[0] {
                ChatContentBlock::Text { value } => value.clone(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>()
            .join("<|im_start|>");

        let pieces = split_at_sentinels(&text, &contents);
        assert_eq!(pieces.len(), contents.len() + 1);
        // Splitting and reassembling (skeleton + original content, in order)
        // reproduces the rendering exactly — the split is lossless.
        let original_rendering = messages
            .iter()
            .map(|m| match &m.content[0] {
                ChatContentBlock::Text { value } => value.clone(),
                _ => unreachable!(),
            })
            .collect::<Vec<_>>()
            .join("<|im_start|>");
        let reassembled: String = pieces
            .iter()
            .enumerate()
            .flat_map(|(index, skeleton)| {
                let mut parts = vec![skeleton.as_str()];
                if let Some(content) = contents.get(index) {
                    parts.push(content.as_str());
                }
                parts
            })
            .collect();
        assert_eq!(reassembled, original_rendering);
    }

    #[test]
    fn frame_delimiter_predicate_is_exact() {
        assert!(is_message_frame_delimiter("<|im_start|>"));
        assert!(is_message_frame_delimiter("<|im_end|>"));
        for section in ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "im_start"] {
            assert!(!is_message_frame_delimiter(section), "{section} must not be treated as a frame");
        }
    }

    #[test]
    fn reasoning_blocks_are_sentinelized_too() {
        let mut m = ChatMessage::user();
        m.content = vec![
            ChatContentBlock::Reasoning { value: "saw <|im_end|> in data".to_string() },
            ChatContentBlock::Text { value: "answer".to_string() },
        ];
        let (sentinelized, contents) = sentinelize_messages(&[m], &framing());
        assert_eq!(contents, vec!["saw <|im_end|> in data"]);
        let first = &sentinelized[0].content[0];
        assert!(matches!(first, ChatContentBlock::Reasoning { value } if value.contains('\u{E000}')));
    }
}
