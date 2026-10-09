use serde::{Deserialize, Serialize};
use uzu::types::{
    basic::{ReasoningEffort, SamplingMethod, SamplingPolicy},
    session::chat::{ChatReply, ChatReplyFinishReason, ChatRole},
};

use super::chart::ChartSpec;

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RunStreamPayload {
    pub run_id: String,
    pub repo_id: String,
    #[serde(default)]
    pub messages: Vec<MsgIn>,
    #[serde(default)]
    pub sampling_policy: Option<SamplingPolicyPayload>,
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub model_chat_naming_enabled: Option<bool>,
    #[serde(default)]
    pub date_time_tool_enabled: Option<bool>,
    #[serde(default)]
    pub chart_tool_enabled: Option<bool>,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all_fields = "camelCase")]
pub enum SamplingDefaults {
    Greedy,
    Stochastic {
        temperature: Option<f64>,
        top_k: Option<i64>,
        top_p: Option<f64>,
        min_p: Option<f64>,
        repetition_penalty: Option<f64>,
        suffix_repetition_length: Option<i64>,
    },
}

impl From<SamplingMethod> for SamplingDefaults {
    fn from(method: SamplingMethod) -> Self {
        match method {
            SamplingMethod::Greedy {} => Self::Greedy,
            SamplingMethod::Stochastic {
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                suffix_repetition_length,
            } => Self::Stochastic {
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                suffix_repetition_length,
            },
        }
    }
}

#[derive(Deserialize, Clone, PartialEq)]
pub struct MsgIn {
    pub role: ChatRole,
    pub content: String,
    #[serde(default, rename = "reasoningContent")]
    pub reasoning_content: Option<String>,
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type")]
pub enum SamplingPolicyPayload {
    Default,
    #[serde(alias = "Argmax")]
    Greedy,
    #[serde(rename_all = "camelCase")]
    Stochastic {
        temperature: Option<f64>,
        top_k: Option<i64>,
        top_p: Option<f64>,
        min_p: Option<f64>,
        #[serde(default)]
        repetition_penalty: Option<f64>,
        #[serde(default)]
        suffix_repetition_length: Option<i64>,
    },
}

#[derive(Serialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct Parsed {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chain_of_thought: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RunStatsSection {
    pub duration: f64,
    pub tokens_count: u32,
    pub tokens_per_second: f64,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TotalStats {
    pub duration: f64,
    pub tokens_count_input: u32,
    pub tokens_count_output: u32,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SessionOutputStats {
    pub prefill_stats: RunStatsSection,
    pub generate_stats: RunStatsSection,
    pub total_stats: TotalStats,
}

#[derive(Serialize, Clone)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum RunEvent {
    #[serde(rename_all = "camelCase")]
    ChatName {
        name: String,
    },
    Transcript {
        items: Vec<TranscriptItem>,
    },
    TranscriptDelta {
        index: usize,
        delta: String,
    },
    #[serde(rename_all = "camelCase")]
    Done {
        text: String,
        stats: SessionOutputStats,
        #[serde(skip_serializing_if = "Option::is_none")]
        finish_reason: Option<ChatReplyFinishReason>,
        #[serde(skip_serializing_if = "Option::is_none")]
        parsed: Option<Parsed>,
        #[serde(skip_serializing_if = "Option::is_none")]
        chat_name: Option<String>,
        transcript: Vec<TranscriptItem>,
    },
    #[serde(rename_all = "camelCase")]
    Error {
        error: String,
    },
}

#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum TranscriptItem {
    Thinking {
        text: String,
        completed: bool,
    },
    Text {
        text: String,
    },
    Chart {
        chart: ChartSpec,
    },
    ToolCall {
        name: String,
        called: bool,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        failed: bool,
    },
}

pub(super) fn map_reply_stats(reply: Option<&ChatReply>) -> SessionOutputStats {
    let stats = reply.map(|r| &r.stats);
    let prefill_duration = stats.and_then(|s| s.time_to_first_token).unwrap_or(0.0);
    let total_duration = stats.map(|s| s.duration).unwrap_or(0.0);
    let generate_duration = (total_duration - prefill_duration).max(0.0);
    let input_tokens = stats.and_then(|s| s.tokens_count_input).unwrap_or(0);
    let output_tokens = stats.and_then(|s| s.tokens_count_output).unwrap_or(0);
    SessionOutputStats {
        prefill_stats: RunStatsSection {
            duration: prefill_duration,
            tokens_count: input_tokens,
            tokens_per_second: stats.and_then(|s| s.prefill_tokens_per_second).unwrap_or(0.0),
        },
        generate_stats: RunStatsSection {
            duration: generate_duration,
            tokens_count: output_tokens,
            tokens_per_second: stats.and_then(|s| s.generate_tokens_per_second).unwrap_or(0.0),
        },
        total_stats: TotalStats {
            duration: total_duration,
            tokens_count_input: input_tokens,
            tokens_count_output: output_tokens,
        },
    }
}

pub(super) fn sampling_policy(payload: &Option<SamplingPolicyPayload>) -> SamplingPolicy {
    match payload {
        None | Some(SamplingPolicyPayload::Default) => SamplingPolicy::Default {},
        Some(SamplingPolicyPayload::Greedy) => SamplingPolicy::Custom {
            method: SamplingMethod::Greedy {},
        },
        Some(SamplingPolicyPayload::Stochastic {
            temperature: Some(temperature),
            ..
        }) if *temperature <= 0.0 => SamplingPolicy::Custom {
            method: SamplingMethod::Greedy {},
        },
        Some(SamplingPolicyPayload::Stochastic {
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            suffix_repetition_length,
        }) => {
            SamplingPolicy::Custom {
                method: SamplingMethod::Stochastic {
                    temperature: *temperature,
                    top_k: *top_k,
                    top_p: *top_p,
                    min_p: *min_p,
                    repetition_penalty: *repetition_penalty,
                    // uzu panics on penalty without suffix length (and on 0).
                    suffix_repetition_length: repetition_penalty
                        .map(|_| suffix_repetition_length.filter(|v| *v > 0).unwrap_or(32)),
                },
            }
        },
    }
}

#[derive(Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct TitleGenPayload {
    pub repo_id: String,
    #[serde(default)]
    pub messages: Vec<MsgIn>,
}

#[cfg(test)]
pub(super) fn test_message(
    role: ChatRole,
    content: &str,
    reasoning: Option<&str>,
) -> MsgIn {
    MsgIn {
        role,
        content: content.to_string(),
        reasoning_content: reasoning.map(str::to_string),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_tool_preferences_remain_distinct_from_explicit_choices() {
        let payload = |extra: serde_json::Value| {
            let mut value = serde_json::json!({ "runId": "run", "repoId": "model" });
            value.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            serde_json::from_value::<RunStreamPayload>(value).unwrap()
        };
        let defaults = payload(serde_json::json!({}));
        assert_eq!(defaults.model_chat_naming_enabled, None);
        assert_eq!(defaults.date_time_tool_enabled, None);
        assert_eq!(defaults.chart_tool_enabled, None);
        for enabled in [true, false] {
            let naming = payload(serde_json::json!({ "modelChatNamingEnabled": enabled }));
            assert_eq!(naming.model_chat_naming_enabled, Some(enabled));
            assert_eq!(naming.date_time_tool_enabled, None);
            assert_eq!(naming.chart_tool_enabled, None);
            let date = payload(serde_json::json!({ "dateTimeToolEnabled": enabled }));
            assert_eq!(date.model_chat_naming_enabled, None);
            assert_eq!(date.date_time_tool_enabled, Some(enabled));
            assert_eq!(date.chart_tool_enabled, None);
            let chart = payload(serde_json::json!({ "chartToolEnabled": enabled }));
            assert_eq!(chart.model_chat_naming_enabled, None);
            assert_eq!(chart.date_time_tool_enabled, None);
            assert_eq!(chart.chart_tool_enabled, Some(enabled));
        }
    }

    #[test]
    fn deserializes_client_reasoning_content() {
        let message: MsgIn = serde_json::from_value(serde_json::json!({
            "role": "assistant",
            "content": "answer",
            "reasoningContent": "reasoning"
        }))
        .expect("message");
        assert_eq!(message.role, ChatRole::Assistant {});
        assert_eq!(message.reasoning_content.as_deref(), Some("reasoning"));
    }

    #[test]
    fn zero_temperature_samples_greedily() {
        let stochastic = |temperature: f64| {
            Some(SamplingPolicyPayload::Stochastic {
                temperature: Some(temperature),
                top_k: Some(40),
                top_p: None,
                min_p: None,
                repetition_penalty: None,
                suffix_repetition_length: None,
            })
        };
        assert!(matches!(
            sampling_policy(&stochastic(0.0)),
            SamplingPolicy::Custom {
                method: SamplingMethod::Greedy {}
            }
        ));
        assert!(matches!(
            sampling_policy(&stochastic(0.7)),
            SamplingPolicy::Custom {
                method: SamplingMethod::Stochastic { .. }
            }
        ));
    }

    #[test]
    fn greedy_accepts_the_previous_argmax_payload_name() {
        for name in ["Greedy", "Argmax"] {
            let payload = serde_json::from_value(serde_json::json!({ "type": name })).unwrap();
            assert!(matches!(payload, SamplingPolicyPayload::Greedy));
            assert!(matches!(
                sampling_policy(&Some(payload)),
                SamplingPolicy::Custom {
                    method: SamplingMethod::Greedy {}
                }
            ));
        }
    }

    #[test]
    fn sampling_defaults_include_the_actual_method_and_keep_disabled_filters_null() {
        let defaults = SamplingDefaults::from(SamplingMethod::Stochastic {
            temperature: None,
            top_k: None,
            top_p: Some(0.95),
            min_p: None,
            repetition_penalty: None,
            suffix_repetition_length: None,
        });
        assert_eq!(
            serde_json::to_value(defaults).unwrap(),
            serde_json::json!({
                "type": "Stochastic", "temperature": null, "topK": null, "topP": 0.95,
                "minP": null, "repetitionPenalty": null, "suffixRepetitionLength": null,
            })
        );
        assert_eq!(
            serde_json::to_value(SamplingDefaults::from(SamplingMethod::Greedy {})).unwrap(),
            serde_json::json!({ "type": "Greedy" })
        );
    }

    #[test]
    fn finish_reason_names_are_the_client_contract() {
        let names: Vec<String> = [
            ChatReplyFinishReason::Stop,
            ChatReplyFinishReason::Length,
            ChatReplyFinishReason::Cancelled,
            ChatReplyFinishReason::ContextLimitReached,
            ChatReplyFinishReason::ToolCalls,
            ChatReplyFinishReason::Rejected,
        ]
        .iter()
        .map(|reason| serde_json::to_value(reason).expect("json").as_str().expect("string").to_string())
        .collect();
        assert_eq!(names, ["Stop", "Length", "Cancelled", "ContextLimitReached", "ToolCalls", "Rejected"]);
    }
}
