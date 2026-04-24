use serde::{Deserialize, Serialize};

#[bindings::export(ClassCloneable)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextToSpeechStats {
    pub text_length: u32,
    pub first_chunk_seconds: f64,
    pub generation_duration: f64,
    pub audio_duration: f64,
}

impl TextToSpeechStats {
    pub fn real_time_factor(&self) -> f64 {
        if self.generation_duration <= 0.0 {
            return 0.0;
        }
        self.audio_duration / self.generation_duration
    }
}
