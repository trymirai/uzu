//! Incremental reply parsing for the VLM completion path.
//!
//! Text models parse their own output through hanashi's encoding stack: a
//! [`TokenStreamParser`] built from the model's parsing config frames `<|im_start|>`
//! messages, tracks `<think>` reasoning sections and `<tool_call>` blocks, and applies
//! the configured transformation to turn markup into structured tool calls. The VLM
//! path renders and decodes on its own pipeline, so this module mirrors the token
//! intake half of [`hanashi::chat::hanashi::HanashiEncodingImpl`] — streaming decode,
//! framing-marker normalization and special-token filtering — against the same parser
//! and the model's own encoding config, without the rendering/history machinery.
//!
//! The encoding config comes from the model resolution (`encoding.json` or the local
//! registry's ChatML/qwen3.5 fallback), so the dialect parsed here is by construction
//! the dialect the text path would parse.

use std::{collections::HashSet, sync::Arc};

use hanashi::chat::{
    EncodingConfig,
    hanashi::{
        config::HanashiConfig,
        messages::streamed::Message as StreamedMessage,
    },
};
use shoji::types::{
    basic::Token,
    session::chat::{ChatContentBlock, ChatRole},
};
use token_stream_parser::{Parser as _, token_stream::TokenStreamParser};
use tokenizers::{Tokenizer, step_decode_stream};

/// The reply-frame content, cumulative over the tokens pushed so far.
#[derive(Debug, Default, Clone)]
pub struct ReplyState {
    pub reasoning: String,
    pub text: String,
    /// Finalized tool calls, in generation order.
    pub tool_calls: Vec<ToolCallValue>,
    /// Unfinished tool-call markup, kept as raw documents (mirrors the chat path's
    /// candidates: an unfinished call must never execute as a partial batch).
    pub tool_call_candidates: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallValue {
    pub name: String,
    pub arguments: String,
}

pub struct VlmReplyParser {
    parser: TokenStreamParser,
    framing_tokens: HashSet<String>,
    pending_framing_tokens: Vec<Token>,
    tokenizer: Arc<Tokenizer>,
    decode_ids: Vec<u32>,
    decode_prefix: String,
    decode_prefix_index: usize,
}

impl VlmReplyParser {
    pub fn new(
        encoding: Option<&HanashiConfig>,
        tokenizer: Arc<Tokenizer>,
    ) -> Result<Self, String> {
        // Local models without an explicit encoding.json resolve to ChatML/qwen3.5,
        // the same fallback the registry applies; a model that ships its own config
        // gets exactly the text path's parsing behavior.
        let config = encoding.cloned().unwrap_or(HanashiConfig::Qwen35);
        let resolved = config.resolve().map_err(|error| error.to_string())?;
        let parser = TokenStreamParser::new(resolved.parsing.clone()).map_err(|error| error.to_string())?;
        let framing_tokens = resolved.parsing.framing_config().tokens.into_iter().collect();
        Ok(Self {
            parser,
            framing_tokens,
            pending_framing_tokens: Vec::new(),
            tokenizer,
            decode_ids: Vec::new(),
            decode_prefix: String::new(),
            decode_prefix_index: 0,
        })
    }

    /// Pushes prompt text (the assistant generation prefix, `<|im_start|>assistant…`)
    /// so the reply frame and any pre-filled `<think>` section exist before sampled
    /// tokens arrive. Section tracking depends on this prefix matching the prefill.
    pub fn push_text(
        &mut self,
        text: &str,
    ) -> Result<(), String> {
        let encoding = self.tokenizer.encode(text, false).map_err(|error| error.to_string())?;
        for token_id in encoding.get_ids() {
            let token = self.resolve_token(*token_id, false)?;
            self.push_normalized_token(token)?;
        }
        self.parser.flush_extraction();
        Ok(())
    }

    /// Pushes one sampled token: streaming decode, framing-marker normalization,
    /// special-token filtering, then extraction. Mirrors hanashi's token intake.
    pub fn push_sampled_token(
        &mut self,
        token_id: u32,
    ) -> Result<(), String> {
        let token = self.resolve_token(token_id, true)?;
        self.push_normalized_token(token)?;
        Ok(())
    }

    /// The current reply frame as cumulative reasoning/text/tool-call blocks.
    pub fn reply_state(&self) -> Option<ReplyState> {
        let value = self.parser.state().value.clone();
        let messages = serde_json::from_value::<Vec<StreamedMessage>>(value).ok()?;
        let message = messages.last()?;
        Some(reply_state_from_message(message))
    }

    fn resolve_token(
        &mut self,
        token_id: u32,
        from_stream: bool,
    ) -> Result<Token, String> {
        let value = if from_stream {
            step_decode_stream(
                &self.tokenizer,
                vec![token_id],
                false,
                &mut self.decode_ids,
                &mut self.decode_prefix,
                &mut self.decode_prefix_index,
            )
            .map_err(|error| error.to_string())?
            .unwrap_or_default()
        } else {
            self.tokenizer.decode(&[token_id], false).map_err(|error| error.to_string())?
        };
        let is_special = self.tokenizer.get_added_vocabulary().is_special_token(&value);
        Ok(Token {
            id: token_id,
            value,
            is_special,
        })
    }

    /// Framing markers may arrive split across sampled tokens; accumulate a suffix
    /// that is a prefix of some marker and only push once it completes. Special
    /// tokens that are not framing markers are dropped, like the text path does.
    fn push_normalized_token(
        &mut self,
        token: Token,
    ) -> Result<(), String> {
        if token.is_special || token.value.is_empty() || self.framing_tokens.contains(&token.value) {
            self.flush_pending()?;
            return self.push_parser_token(&token);
        }
        loop {
            let text: String = self
                .pending_framing_tokens
                .iter()
                .map(|token| token.value.as_str())
                .chain(std::iter::once(token.value.as_str()))
                .collect();
            let marker = self
                .framing_tokens
                .iter()
                .filter(|marker| {
                    !self.tokenizer.get_added_vocabulary().is_special_token(marker) && marker.starts_with(&text)
                })
                .filter_map(|marker| self.tokenizer.token_to_id(marker).map(|id| (marker.len() == text.len(), id)))
                .min_by_key(|(complete, _)| !complete);
            match marker {
                Some((true, id)) => {
                    self.pending_framing_tokens.clear();
                    // Normalize the parser view only; the sampler keeps its own ids.
                    let marker = self.resolve_token(id, false)?;
                    return self.push_parser_token(&marker);
                },
                Some((false, _)) => {
                    self.pending_framing_tokens.push(token.clone());
                    return Ok(());
                },
                None if self.pending_framing_tokens.is_empty() => return self.push_parser_token(&token),
                None => {
                    self.flush_pending()?;
                },
            }
        }
    }

    fn flush_pending(&mut self) -> Result<(), String> {
        if self.pending_framing_tokens.is_empty() {
            return Ok(());
        }
        for token in std::mem::take(&mut self.pending_framing_tokens) {
            self.push_parser_token(&token)?;
        }
        self.parser.flush_extraction();
        Ok(())
    }

    fn push_parser_token(
        &mut self,
        token: &Token,
    ) -> Result<(), String> {
        // Special tokens that are not framing markers carry no parseable content.
        if token.is_special && !self.framing_tokens.contains(&token.value) {
            return Ok(());
        }
        let parser_token = token_stream_parser::types::Token {
            id: token.id,
            value: token.value.clone(),
            is_special: token.is_special,
        };
        self.parser.push(&parser_token).map_err(|error| error.to_string())
    }
}

/// Converts the streamed message into reply blocks, mirroring the chat path's
/// section semantics: finalized calls, unfinished markup as candidates.
fn reply_state_from_message(message: &StreamedMessage) -> ReplyState {
    let mut state = ReplyState::default();
    let role = ChatRole::Assistant {};
    let blocks = match &message.content {
        Some(content) => content.clone().blocks(&role),
        None => Vec::new(),
    };
    for block in blocks {
        match block {
            ChatContentBlock::Text {
                value,
            } => state.text.push_str(&value),
            ChatContentBlock::Reasoning {
                value,
            } => state.reasoning.push_str(&value),
            ChatContentBlock::ToolCall {
                value,
            } => state.tool_calls.push(ToolCallValue {
                name: value.name,
                arguments: value.arguments.json,
            }),
            ChatContentBlock::ToolCallCandidate {
                value,
                ..
            } => {
                // Candidates carry partial markup as a JSON string document or an object.
                let raw = serde_json::from_str::<String>(&value.json).unwrap_or_else(|_| value.json.clone());
                state.tool_call_candidates.push(raw);
            },
            _ => {},
        }
    }
    state
}

/// Turns a model encoding entry into the hanashi chat config it names, or None when
/// the model uses a non-hanashi encoding (no reply parsing for the VLM path then).
pub fn reply_encoding_from_json(encoding_json: Option<&str>) -> Option<HanashiConfig> {
    let json = encoding_json?;
    match serde_json::from_str::<EncodingConfig>(json) {
        Ok(EncodingConfig::Hanashi {
            config,
        }) => Some(config),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Env-gated: LFM3_VL_MODEL points at the bundle (tokenizer.json + weights not needed).
    #[test]
    fn parses_lfm3_vl_xml_tool_calls() {
        let Ok(bundle) = std::env::var("LFM3_VL_MODEL") else {
            eprintln!("skipped: LFM3_VL_MODEL not set");
            return;
        };
        let tokenizer = Arc::new(Tokenizer::from_file(std::path::Path::new(&bundle).join("tokenizer.json")).unwrap());
        let mut parser = VlmReplyParser::new(Some(&HanashiConfig::Lfm3Vl), tokenizer.clone()).unwrap();
        parser.push_text("<|im_start|>assistant\n<think>\n\n</think>\n\n").unwrap();
        let completion = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        for id in tokenizer.encode(completion, false).unwrap().get_ids() {
            parser.push_sampled_token(*id).unwrap();
        }
        let state = parser.reply_state().unwrap();
        assert_eq!(state.text.trim(), "");
        assert_eq!(state.tool_calls.len(), 1, "state={state:?}");
        assert_eq!(state.tool_calls[0].name, "get_weather");
        assert_eq!(state.tool_calls[0].arguments, r#"{"city":"Paris"}"#);
    }
}
