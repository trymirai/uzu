use serde::{Deserialize, Serialize};
use serde_json::Value;
use shoji::types::session::chat::{ChatReplyJoulesPerToken, ChatReplyStats};

/// Reply stats plus the metrics `ChatReplyStats` exposes as getters, which serde
/// would otherwise leave out of the reported payload.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelemetryStats {
    #[serde(flatten)]
    pub stats: ChatReplyStats,
    pub input_joules_per_token: Option<ChatReplyJoulesPerToken>,
    pub output_joules_per_token: Option<ChatReplyJoulesPerToken>,
}

impl From<ChatReplyStats> for TelemetryStats {
    fn from(stats: ChatReplyStats) -> Self {
        Self {
            input_joules_per_token: stats.input_joules_per_token(),
            output_joules_per_token: stats.output_joules_per_token(),
            stats,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_name", content = "payload", rename_all = "snake_case")]
pub enum TelemetryEvent {
    ModelDownloadStarted {
        model_id: String,
    },
    ModelDownloadFinished {
        model_id: String,
    },
    ModelInferenceStarted {
        model_id: String,
    },
    ModelInferenceFinished {
        model_id: String,
        stats: TelemetryStats,
    },
    ModelInferenceFailed {
        error: Value,
    },
}
