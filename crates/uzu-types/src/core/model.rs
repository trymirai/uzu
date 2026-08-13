use enumset::EnumSet;

use crate::{
    core::{
        backend::InferenceBackend,
        content::{ContentPart, ContentType},
        state::InferenceState,
    },
    model::ModelCapability,
};

/// A model that performs inference through an [`InferenceBackend`].
///
/// Implementations describe the model's capability and supported content types,
/// provide an initial inference state, and execute inference for a sequence of input content parts.
pub trait InferenceModel {
    type Backend: InferenceBackend;

    fn capability() -> ModelCapability;

    fn supported_input_types() -> EnumSet<ContentType>;

    fn supported_output_types() -> EnumSet<ContentType>;

    fn create_empty_state() -> Box<dyn InferenceState>;

    fn infer(
        input: &[ContentPart],
        state: Box<dyn InferenceState>,
    ) -> <<Self as InferenceModel>::Backend as InferenceBackend>::Output;
}
