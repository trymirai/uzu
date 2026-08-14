use enumset::EnumSet;

use crate::{
    core::{InferenceOutput, content::ContentKind, state::InferenceState},
    model::ModelCapability,
    session::chat::ChatMessage,
};

/// Implementations describe the model's capability and supported content types,
/// provide an initial inference state, and execute inference for a sequence of input content parts.
pub trait InferenceModel {
    fn capability(&self) -> ModelCapability;

    fn supported_input_types(&self) -> EnumSet<ContentKind>;

    fn supported_output_types(&self) -> EnumSet<ContentKind>;

    fn create_empty_state(&self) -> Box<dyn InferenceState>;

    fn reply(
        &self,
        input: &[ChatMessage],
        state: Box<dyn InferenceState>,
    ) -> Box<dyn InferenceOutput>;
}
