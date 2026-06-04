use serde::{Deserialize, Serialize};
use shoji::types::session::chat::ChatReplyStats;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TelemetryEvent {
    ModelDownloadFinished {
        model_id: String,
    },
    SessionGenerationFinished {
        model_id: String,
        stats: ChatReplyStats,
    },
}
