use shoji::types::{
    basic::TokenId,
    session::chat::{ChatContentBlock, ChatMessage, ChatRole},
};
use token_stream_parser::{Parser as _, reduction::ReductionParserSection, token_stream::TokenStreamParser};

use super::{Error, HanashiEncodingImpl, config::HanashiConfig};

impl HanashiEncodingImpl {
    /// Append a rendered continuation without replacing any sampled prefix token IDs.
    /// Returns None without changing state when the history or message boundary is incompatible.
    pub fn try_append(
        &mut self,
        messages: &[ChatMessage],
    ) -> Result<Option<Vec<TokenId>>, Error> {
        let previous_len = self.state.messages.len();
        if messages.len() <= previous_len
            || !messages.last().is_some_and(|message| matches!(message.role, ChatRole::User {} | ChatRole::Tool {}))
            || !self.state.messages.last().is_some_and(|message| {
                matches!(message.role, ChatRole::Assistant {})
                    && !message.content.iter().any(|block| matches!(block, ChatContentBlock::ToolCallCandidate { .. }))
            })
        {
            return Ok(None);
        }

        let Some((group, close)) = Self::completed_boundary(&self.parser) else {
            return Ok(None);
        };

        if !self
            .state
            .tokens
            .last()
            .is_some_and(|token| token.is_special && token.id == close.id && token.value == close.value)
        {
            return Ok(None);
        }

        let boundary = (group.to_string(), close.value.clone());
        let previous = self.fill_default_content(&self.state.messages)?;
        let previous_text = self.render_messages(&previous, false)?;
        let Some(prefix_end) = self.rendered_boundary(&previous_text, &boundary)? else {
            return Ok(None);
        };

        let messages = self.fill_default_content(messages)?;
        let next_text = self.render_messages(&messages, true)?;
        let Some(suffix) = next_text.strip_prefix(&previous_text[..prefix_end]) else {
            return Ok(None);
        };

        let suffix_ids = self.tokenize(suffix)?;
        if suffix_ids.is_empty() {
            return Ok(None);
        }

        // Build the candidate separately: parsing/validation errors must not damage the
        // encoding that still describes the live backend. Replay the exact cached tokens.
        let mut candidate = self.empty_copy()?;
        for message in &messages {
            candidate.validator.validate_next(&message.role)?;
        }
        if messages
            .iter()
            .any(|message| message.content.iter().any(|block| matches!(block, ChatContentBlock::Tools { .. })))
        {
            candidate.parser.set_variable("tools", serde_json::Value::Bool(true));
        }
        candidate.state.messages = messages;
        for token in &self.state.tokens {
            candidate.push_token_to_parser(token, true)?;
        }
        candidate.state.tokens = self.state.tokens.clone();
        candidate.tokenizer_decode_ids = self.tokenizer_decode_ids.clone();
        candidate.tokenizer_decode_prefix = self.tokenizer_decode_prefix.clone();
        candidate.tokenizer_decode_prefix_index = self.tokenizer_decode_prefix_index;
        for id in &suffix_ids {
            let token = candidate.resolve_token(*id, true)?;
            candidate.push_token_to_parser(&token, true)?;
            candidate.state.tokens.push(token);
        }
        candidate.parser.flush_extraction();
        candidate.update_messages_from_parser_state()?;
        *self = candidate;

        Ok(Some(suffix_ids))
    }

    fn empty_copy(&self) -> Result<Self, Error> {
        Self::new(
            HanashiConfig::Custom {
                config: self.config.clone(),
            },
            self.tokenizer.clone(),
        )
    }

    /// Locate the canonical counterpart of the sampled message's closing marker.
    /// Template-owned whitespace after that marker belongs to the new suffix.
    fn rendered_boundary(
        &self,
        text: &str,
        boundary: &(String, String),
    ) -> Result<Option<usize>, Error> {
        let mut rendered = self.empty_copy()?;
        let mut decoded = String::new();
        let mut end = None;
        for id in self.tokenize(text)? {
            let token = rendered.resolve_token(id, true)?;
            decoded.push_str(&token.value);
            rendered.push_token_to_parser(&token, true)?;
            if let Some((group, close)) = Self::completed_boundary(&rendered.parser)
                && group == boundary.0
                && close.value == boundary.1
                && token.is_special
                && token.id == close.id
                && token.value == close.value
            {
                end = Some(decoded.len());
            }
        }
        // A normalizing tokenizer or a template without the corresponding completed
        // message cannot supply a safe byte boundary for splicing.
        Ok(end.filter(|&end| decoded == text && text[end..].chars().all(char::is_whitespace)))
    }

    /// A closed root group, not merely a completed section inside an open message.
    fn completed_boundary(parser: &TokenStreamParser) -> Option<(&str, &token_stream_parser::types::Token)> {
        match parser.reduction().state().sections.last()? {
            ReductionParserSection::Group {
                name,
                close: Some(close),
                finished: true,
                ..
            } => Some((name, close)),
            _ => None,
        }
    }
}
