use openai_harmony::chat::ReasoningEffort as ExternalReasoningEffort;

use crate::chat::{encoding::harmony::bridging::Error, types::ReasoningEffort};

impl TryFrom<ReasoningEffort> for ExternalReasoningEffort {
    type Error = Error;

    fn try_from(reasoning_effort: ReasoningEffort) -> Result<ExternalReasoningEffort, Error> {
        match reasoning_effort {
            ReasoningEffort::Low => Ok(ExternalReasoningEffort::Low),
            ReasoningEffort::Medium => Ok(ExternalReasoningEffort::Medium),
            ReasoningEffort::High => Ok(ExternalReasoningEffort::High),
            other => Err(Error::UnsupportedReasoningEffort {
                reasoning_effort: other,
            }),
        }
    }
}

impl From<ExternalReasoningEffort> for ReasoningEffort {
    fn from(reasoning_effort: ExternalReasoningEffort) -> ReasoningEffort {
        match reasoning_effort {
            ExternalReasoningEffort::Low => ReasoningEffort::Low,
            ExternalReasoningEffort::Medium => ReasoningEffort::Medium,
            ExternalReasoningEffort::High => ReasoningEffort::High,
        }
    }
}
