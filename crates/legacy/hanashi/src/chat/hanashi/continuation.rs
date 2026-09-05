use shoji::types::{
    basic::TokenId,
    session::chat::{ChatContentBlock, ChatMessage, ChatRole},
};
use token_stream_parser::{Parser as _, reduction::ReductionParserSection};
use tokenizers::Tokenizer;

use super::{Error, HanashiEncodingImpl, config::HanashiConfig, ordering::Validator};
use crate::chat::{SynchronizationError, SynchronizationResult};

const MESSAGE_END: &str = "<|im_end|>";

#[derive(Clone, Copy)]
pub(super) enum ContinuationPolicy {
    Qwen35 {
        message_end_id: TokenId,
    },
}

impl ContinuationPolicy {
    pub(super) fn new(
        config: &HanashiConfig,
        tokenizer: &Tokenizer,
    ) -> Option<Self> {
        if !matches!(config, HanashiConfig::Qwen35) || !tokenizer.get_added_vocabulary().is_special_token(MESSAGE_END) {
            return None;
        }
        let message_end_id = tokenizer.token_to_id(MESSAGE_END)?;
        if tokenizer.decode(&[message_end_id], false).ok()?.as_str() != MESSAGE_END {
            return None;
        }
        Some(Self::Qwen35 {
            message_end_id,
        })
    }
}

pub(super) struct CompletedGeneration {
    canonical_prefix: String,
}

impl HanashiEncodingImpl {
    pub fn supports_continuation(&self) -> bool {
        self.continuation_policy.is_some()
    }

    /// Record authoritative history only after the owner has finalized a successful backend stream.
    pub fn record_completion(
        &mut self,
        messages: Vec<ChatMessage>,
    ) -> Result<bool, Error> {
        self.completed_generation = None;
        let Some(ContinuationPolicy::Qwen35 {
            message_end_id,
        }) = self.continuation_policy
        else {
            return Ok(false);
        };
        if !self.has_completed_assistant(message_end_id)
            || !messages.last().is_some_and(|message| matches!(message.role, ChatRole::Assistant {}))
            || !messages.iter().all(is_supported_message)
        {
            return Ok(false);
        }

        let rendered_messages = self.fill_default_content(&messages)?;
        let mut canonical_prefix = self.render_messages(&rendered_messages, false)?;
        if !canonical_prefix.ends_with("<|im_end|>\n") {
            return Ok(false);
        }
        // The stop token was emitted, but the template's following newline has not been consumed.
        canonical_prefix.pop();
        self.state.messages = messages;
        self.completed_generation = Some(CompletedGeneration {
            canonical_prefix,
        });
        Ok(true)
    }

    /// Append a certified continuation. An error requires the owner to reset encoding and backend state together.
    pub fn try_append(
        &mut self,
        messages: &[ChatMessage],
    ) -> Result<Option<Vec<TokenId>>, Error> {
        let previous_len = self.state.messages.len();
        if self.completed_generation.is_none()
            || messages.len() <= previous_len
            || !messages.starts_with(&self.state.messages)
            || !messages.last().is_some_and(|message| matches!(message.role, ChatRole::User {} | ChatRole::Tool {}))
            || !messages[previous_len..].iter().all(is_supported_append)
        {
            return Ok(None);
        }

        let rendered_messages = self.fill_default_content(messages)?;
        let mut validator = Validator::new(self.config.ordering.clone());
        for message in &rendered_messages {
            validator.validate_next(&message.role)?;
        }
        let rendered = self.render_messages(&rendered_messages, true)?;
        let Some(completed) = &self.completed_generation else {
            return Ok(None);
        };
        let Some(suffix) = rendered.strip_prefix(&completed.canonical_prefix) else {
            return Ok(None);
        };
        let token_ids = self.tokenize(suffix)?;
        if token_ids.is_empty() {
            return Ok(None);
        }

        // Keep the sampled token ledger and incremental decoder; only new input is staged and parsed.
        self.completed_generation = None;
        self.state.messages.extend(rendered_messages.into_iter().skip(previous_len));
        self.validator = validator;
        let staged_len = self.state.messages.len();
        let synchronization = self.push_prompt_tokens(&token_ids)?;
        if synchronization != SynchronizationResult::Inserted
            || self.state.messages.len() != staged_len + 1
            || !self.state.messages.last().is_some_and(|message| matches!(message.role, ChatRole::Assistant {}))
        {
            return Err(SynchronizationError::Desynchronization.into());
        }
        Ok(Some(token_ids))
    }

    fn has_completed_assistant(
        &self,
        message_end_id: TokenId,
    ) -> bool {
        let Some(last_token) = self.state.tokens.last() else {
            return false;
        };
        if last_token.id != message_end_id || !last_token.is_special || last_token.value != MESSAGE_END {
            return false;
        }
        if !self.state.messages.last().is_some_and(|message| {
            matches!(message.role, ChatRole::Assistant {})
                && !message.content.iter().any(|block| matches!(block, ChatContentBlock::ToolCallCandidate { .. }))
        }) {
            return false;
        }
        if self
            .parser
            .state()
            .value
            .as_array()
            .and_then(|messages| messages.last())
            .and_then(|message| message.get("role").and_then(serde_json::Value::as_str))
            != Some("assistant")
        {
            return false;
        }
        let Some(ReductionParserSection::Group {
            name,
            open: Some(open),
            close: Some(close),
            finished: true,
            sections,
        }) = self.parser.reduction().state().sections.last()
        else {
            return false;
        };
        name == "message"
            && open.value == "<|im_start|>"
            && open.is_special
            && close.id == message_end_id
            && close.value == MESSAGE_END
            && close.is_special
            && sections.iter().all(has_closed_bounded_groups)
    }
}

fn has_closed_bounded_groups(section: &ReductionParserSection) -> bool {
    let ReductionParserSection::Group {
        name,
        close,
        finished,
        sections,
        ..
    } = section
    else {
        return true;
    };
    let expected_close = match name.as_str() {
        "role" | "content" => None,
        "reasoning" => Some("</think>"),
        "tool_call" => Some("</tool_call>"),
        "tool_call_result" => Some("</tool_response>"),
        _ => return false,
    };
    *finished
        && expected_close.is_none_or(|expected| close.as_ref().is_some_and(|token| token.value == expected))
        && sections.iter().all(has_closed_bounded_groups)
}

fn is_supported_message(message: &ChatMessage) -> bool {
    !matches!(message.role, ChatRole::Custom { .. })
        && message.content.iter().all(|block| {
            matches!(
                block,
                ChatContentBlock::Text { .. }
                    | ChatContentBlock::Reasoning { .. }
                    | ChatContentBlock::ToolCall { .. }
                    | ChatContentBlock::ToolCallResult { .. }
                    | ChatContentBlock::Tools { .. }
                    | ChatContentBlock::ReasoningEffort { .. }
            )
        })
}

fn is_supported_append(message: &ChatMessage) -> bool {
    is_supported_message(message)
        && matches!(message.role, ChatRole::User {} | ChatRole::Assistant {} | ChatRole::Tool {})
        && !message
            .content
            .iter()
            .any(|block| matches!(block, ChatContentBlock::Tools { .. } | ChatContentBlock::ReasoningEffort { .. }))
}
