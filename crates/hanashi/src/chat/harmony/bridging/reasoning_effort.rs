use openai_harmony::chat::ReasoningEffort as ExternalReasoningEffort;
use shoji::types::session::chat::ChatReasoningEffort;

use crate::chat::harmony::bridging::{Error, FromHarmony, ToHarmony};

impl ToHarmony for ChatReasoningEffort {
    type Output = ExternalReasoningEffort;

    fn to_harmony(self) -> Result<Self::Output, Error> {
        match self {
            ChatReasoningEffort::Low => Ok(ExternalReasoningEffort::Low),
            ChatReasoningEffort::Medium => Ok(ExternalReasoningEffort::Medium),
            ChatReasoningEffort::High => Ok(ExternalReasoningEffort::High),
            other => Err(Error::UnsupportedReasoningEffort {
                reasoning_effort: other,
            }),
        }
    }
}

impl FromHarmony for ChatReasoningEffort {
    type Input = ExternalReasoningEffort;

    fn from_harmony(input: Self::Input) -> Self {
        match input {
            ExternalReasoningEffort::Low => ChatReasoningEffort::Low,
            ExternalReasoningEffort::Medium => ChatReasoningEffort::Medium,
            ExternalReasoningEffort::High => ChatReasoningEffort::High,
        }
    }
}
