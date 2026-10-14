//! The same declarative chat encodings used by the desktop app. Inference stays
//! in LanguageModel; this layer owns templates, reasoning and tool transcripts.
use hanashi::{
    Encoding as _,
    chat::{EncodingConfig, hanashi::HanashiEncodingImpl},
};
use serde::Serialize;
use shoji::types::{
    basic::{ReasoningEffort, ToolDescription, ToolNamespace},
    session::chat::{ChatContentBlock, ChatMessage, ChatModelCapabilities, ChatRole},
};

use crate::chat::{Message, RunParams};

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub(crate) enum ReasoningSupport {
    Unsupported,
    AlwaysOn,
    Toggle {
        default_effort: ReasoningEffort,
    },
    Levels {
        efforts: Vec<ReasoningEffort>,
        #[serde(skip_serializing_if = "Option::is_none")]
        default_effort: Option<ReasoningEffort>,
    },
}

impl ReasoningSupport {
    pub fn effective(
        &self,
        requested: Option<&str>,
    ) -> Option<ReasoningEffort> {
        let requested = requested.and_then(|value| value.parse().ok());
        match self {
            Self::Unsupported | Self::AlwaysOn => None,
            Self::Toggle {
                ..
            } => requested.filter(|effort| *effort == ReasoningEffort::Disabled),
            Self::Levels {
                efforts,
                ..
            } => requested.filter(|effort| *effort != ReasoningEffort::Default && efforts.contains(effort)),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct EncodingSupport {
    pub reasoning: ReasoningSupport,
    pub supports_tools: bool,
}

pub(crate) fn inspect(config: &EncodingConfig) -> Result<EncodingSupport, String> {
    let capabilities = config.capabilities().map_err(|e| e.to_string())?;
    let default = config.default_reasoning_effort().map_err(|e| e.to_string())?;
    Ok(EncodingSupport {
        reasoning: classify(&capabilities, default),
        supports_tools: capabilities.supports_tools,
    })
}

fn classify(
    capabilities: &ChatModelCapabilities,
    default: Option<ReasoningEffort>,
) -> ReasoningSupport {
    use ReasoningEffort::*;
    if !capabilities.supports_reasoning {
        return ReasoningSupport::Unsupported;
    }
    if capabilities.reasoning_efforts.iter().any(|effort| matches!(effort, Low | Medium | High | XHigh)) {
        let efforts: Vec<_> = [Disabled, Low, Medium, High, XHigh]
            .into_iter()
            .filter(|effort| capabilities.reasoning_efforts.contains(effort))
            .collect();
        let default_effort = default.filter(|effort| efforts.contains(effort));
        ReasoningSupport::Levels {
            efforts,
            default_effort,
        }
    } else if capabilities.supports_disable_reasoning {
        if default == Some(Default) && capabilities.reasoning_efforts.contains(&Default) {
            ReasoningSupport::Toggle {
                default_effort: Default,
            }
        } else {
            ReasoningSupport::Levels {
                efforts: vec![Disabled],
                default_effort: default.filter(|effort| *effort == Disabled),
            }
        }
    } else {
        ReasoningSupport::AlwaysOn
    }
}

pub(crate) fn messages(
    raw: &[Message],
    effort: Option<ReasoningEffort>,
    tools: Vec<ToolDescription>,
) -> Result<Vec<ChatMessage>, String> {
    let mut messages = raw
        .iter()
        .map(|message| {
            let role = match message.role.as_str() {
                "system" => ChatRole::System {},
                "user" => ChatRole::User {},
                "assistant" => ChatRole::Assistant {},
                _ => return Err("Unsupported chat role".into()),
            };
            let mut built = ChatMessage::for_role(role).with_text(message.content.clone());
            if message.role == "assistant"
                && let Some(reasoning) = &message.reasoning_content
            {
                built = built.with_reasoning(reasoning.clone());
            }
            Ok(built)
        })
        .collect::<Result<Vec<_>, String>>()?;
    if let Some(effort) = effort {
        if let Some(system) = messages.iter_mut().find(|message| message.role == (ChatRole::System {})) {
            *system = system.with_reasoning_effort(effort);
        } else {
            messages.insert(0, ChatMessage::system().with_reasoning_effort(effort));
        }
    }
    if !tools.is_empty() {
        let at = usize::from(messages.first().is_some_and(|message| message.role == (ChatRole::System {})));
        messages.insert(
            at,
            ChatMessage::developer().with_tool_namespaces(vec![ToolNamespace {
                name: "functions".into(),
                description: None,
                tools,
            }]),
        );
    }
    Ok(messages)
}

pub(crate) struct EncodedState {
    pub encoder: HanashiEncodingImpl,
    pub visible: Vec<Message>,
    pub effort: Option<ReasoningEffort>,
    pub tools: Vec<ToolDescription>,
}

impl EncodedState {
    pub fn prepare(
        &mut self,
        params: &RunParams,
        effort: Option<ReasoningEffort>,
    ) -> Result<Vec<u64>, String> {
        // Preserve internal tool results while the visible history still matches.
        let continuation = !self.visible.is_empty()
            && params.messages.starts_with(&self.visible)
            && params.messages.len() > self.visible.len()
            && self.effort == effort
            && self.tools == params.tools;
        let all = if continuation {
            let mut all = self.encoder.state().messages.clone();
            all.extend(messages(&params.messages[self.visible.len()..], None, vec![])?);
            all
        } else {
            messages(&params.messages, effort, params.tools.clone())?
        };
        self.visible.clear();
        self.effort = effort;
        self.tools = params.tools.clone();
        self.prepare_messages(all)
    }

    pub fn prepare_messages(
        &mut self,
        all: Vec<ChatMessage>,
    ) -> Result<Vec<u64>, String> {
        if self.encoder.try_append(&all).map_err(|e| e.to_string())?.is_none() {
            self.encoder.reset().map_err(|e| e.to_string())?;
            self.encoder.encode(all).map_err(|e| e.to_string())?;
        }
        Ok(self.encoder.state().tokens.iter().map(|token| u64::from(token.id)).collect())
    }

    pub fn reply(&self) -> Option<&ChatMessage> {
        self.encoder.state().messages.last().filter(|message| message.role == (ChatRole::Assistant {}))
    }
}

pub(crate) fn has_tool_candidates(message: &ChatMessage) -> bool {
    message
        .content
        .iter()
        .any(|block| matches!(block, ChatContentBlock::ToolCall { .. } | ChatContentBlock::ToolCallCandidate { .. }))
}

pub(crate) fn sanitize(text: &str) -> String {
    text.chars()
        .filter(|&c| {
            !((c.is_control() && !matches!(c, '\n' | '\r' | '\t')) || matches!(c, '\u{FFFD}' | '\u{FFFC}' | '\u{FEFF}'))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn catalog_capabilities_match_desktop_encodings() {
        for (name, tools, reasoning) in [
            ("qwen3.5", true, "toggle"),
            ("lfm2.5-instruct", true, "unsupported"),
            ("lfm2.5-thinking", true, "alwaysOn"),
        ] {
            let config = serde_json::from_value(serde_json::json!({"type":"hanashi", "name":name})).unwrap();
            let support = serde_json::to_value(inspect(&config).unwrap()).unwrap();
            assert_eq!(support["supportsTools"], tools);
            assert_eq!(support["reasoning"]["kind"], reasoning);
        }
    }
    #[test]
    fn parameters_preserve_history_reasoning_and_tools_in_their_declared_roles() {
        let params: RunParams = serde_json::from_value(serde_json::json!({"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"reply","reasoningContent":"reason"}], "tools":[{"type":"function","function":{"name":"test","description":"test","parameters":{"type":"object"}}}]})).unwrap();
        let built = messages(&params.messages, Some(ReasoningEffort::Disabled), params.tools).unwrap();
        assert_eq!(built[0].reasoning_effort(), Some(ReasoningEffort::Disabled));
        assert_eq!(built[1].role, ChatRole::Developer {});
        assert_eq!(built[1].tool_namespaces()[0].tools.len(), 1);
        assert_eq!(built[3].reasoning().as_deref(), Some("reason"));
    }
    fn tokenizer() -> std::sync::Arc<tokenizers::Tokenizer> {
        use tokenizers::{AddedToken, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};
        let vocab: tokenizers::models::bpe::Vocab =
            ByteLevel::alphabet().into_iter().enumerate().map(|(id, ch)| (ch.to_string(), id as u32)).collect();
        let mut tokenizer = tokenizers::Tokenizer::new(BPE::builder().vocab_and_merges(vocab, vec![]).build().unwrap());
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
        tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        tokenizer.add_special_tokens(
            &[
                "<|im_start|>",
                "<|im_end|>",
                "<think>",
                "</think>",
                "<|tool_call_start|>",
                "<|tool_call_end|>",
                "<tool_call>",
                "</tool_call>",
                "<tool_response>",
                "</tool_response>",
            ]
            .into_iter()
            .map(|text| AddedToken::from(text, true))
            .collect::<Vec<_>>(),
        );
        tokenizer.add_tokens(
            &["assistant", "system", "user", "developer", "tool"]
                .into_iter()
                .map(|text| AddedToken::from(text, false))
                .collect::<Vec<_>>(),
        );
        std::sync::Arc::new(tokenizer)
    }

    #[test]
    fn desktop_encodings_parse_reasoning_and_real_tool_calls_without_string_heuristics() {
        use hanashi::chat::hanashi::config::HanashiConfig;
        for (config, generated) in [
            (
                HanashiConfig::Qwen35,
                "<tool_call>\n<function=get_current_date_time>\n</function>\n</tool_call><|im_end|>",
            ),
            (HanashiConfig::Lfm25Instruct, "<|tool_call_start|>[get_current_date_time()]<|tool_call_end|><|im_end|>"),
        ] {
            let tokenizer = tokenizer();
            let effort = config.capabilities().unwrap().supports_disable_reasoning;
            let mut encoder = HanashiEncodingImpl::new(config, tokenizer.clone()).unwrap();
            let mut input = vec![ChatMessage::user().with_text("hello".into())];
            if effort {
                input.insert(0, ChatMessage::system().with_reasoning_effort(ReasoningEffort::Disabled));
            }
            encoder.encode(input).unwrap();
            encoder.decode(tokenizer.encode(generated, false).unwrap().get_ids().to_vec()).unwrap();
            let reply = encoder.state().messages.last().unwrap();
            assert_eq!(reply.tool_calls()[0].name, "get_current_date_time");
            assert_eq!(reply.text().unwrap_or_default(), "");
        }
        let tokenizer = tokenizer();
        let mut encoder = HanashiEncodingImpl::new(HanashiConfig::Qwen35, tokenizer.clone()).unwrap();
        encoder.encode(vec![ChatMessage::user().with_text("hello".into())]).unwrap();
        encoder
            .decode(
                tokenizer
                    .encode("Considering carefully</think>\n\nAnswer<|im_end|>", false)
                    .unwrap()
                    .get_ids()
                    .to_vec(),
            )
            .unwrap();
        let reply = encoder.state().messages.last().unwrap();
        assert_eq!(reply.reasoning().unwrap().trim(), "Considering carefully");
        assert_eq!(reply.text().unwrap(), "Answer");
    }
}
