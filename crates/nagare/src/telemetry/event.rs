use serde::{Deserialize, Serialize};
use serde_json::Value;
use shoji::types::session::chat::ChatReplyStats;

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
        stats: ChatReplyStats,
    },
    ModelInferenceFailed {
        error: Value,
    },
}
