use serde::{Deserialize, Serialize};

use crate::types::{basic::PcmBatch, session::text_to_speech::TextToSpeechStats};

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextToSpeechOutput {
    pub pcm_batch: PcmBatch,
    pub stats: TextToSpeechStats,
}
