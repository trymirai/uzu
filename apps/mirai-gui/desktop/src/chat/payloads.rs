use serde::{Deserialize, Serialize};
use uzu::types::{
    basic::{ReasoningEffort, SamplingMethod, SamplingPolicy},
    session::chat::{ChatReply, ChatReplyFinishReason, ChatRole},
};

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
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SamplingDefaults {
    pub temperature: Option<f64>,
    pub top_k: Option<i64>,
    pub top_p: Option<f64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub suffix_repetition_length: Option<i64>,
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
    Argmax,
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
    Chunk {
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        parsed: Option<Parsed>,
    },
    #[serde(rename_all = "camelCase")]
    Done {
        text: String,
        stats: SessionOutputStats,
        #[serde(skip_serializing_if = "Option::is_none")]
        finish_reason: Option<ChatReplyFinishReason>,
        #[serde(skip_serializing_if = "Option::is_none")]
        parsed: Option<Parsed>,
    },
    #[serde(rename_all = "camelCase")]
    Error {
        error: String,
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
        Some(SamplingPolicyPayload::Argmax) => SamplingPolicy::Custom {
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
