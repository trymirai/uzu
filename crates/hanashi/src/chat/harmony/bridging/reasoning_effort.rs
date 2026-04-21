use openai_harmony::chat::ReasoningEffort as ExternalReasoningEffort;
use shoji::types::session::chat::ReasoningEffort;

use crate::chat::harmony::bridging::{Error, FromHarmony, ToHarmony};

impl ToHarmony for ReasoningEffort {
    type Output = ExternalReasoningEffort;

    fn to_harmony(self) -> Result<Self::Output, Error> {
        match self {
            ReasoningEffort::Low => Ok(ExternalReasoningEffort::Low),
            ReasoningEffort::Medium => Ok(ExternalReasoningEffort::Medium),
            ReasoningEffort::High => Ok(ExternalReasoningEffort::High),
            other => Err(Error::UnsupportedReasoningEffort {
                reasoning_effort: other,
            }),
        }
    }
}

impl FromHarmony for ReasoningEffort {
    type Input = ExternalReasoningEffort;

    fn from_harmony(input: Self::Input) -> Self {
        match input {
            ExternalReasoningEffort::Low => ReasoningEffort::Low,
            ExternalReasoningEffort::Medium => ReasoningEffort::Medium,
            ExternalReasoningEffort::High => ReasoningEffort::High,
        }
    }
}
