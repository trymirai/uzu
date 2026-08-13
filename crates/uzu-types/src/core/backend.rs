use crate::{
    core::{model::InferenceModel, output::InferenceOutput},
    model::ModelSource,
};

/// A backend that loads models and provides their inference output type.
///
/// Implementations connect a concrete [`InferenceModel`] and [`InferenceOutput`] to the runtime or service
/// responsible for executing inference.
pub trait InferenceBackend {
    type Model: InferenceModel;
    type Output: InferenceOutput;

    fn load_model(desc: &ModelSource) -> Self::Model;

    fn identifier() -> String;

    fn version() -> String;
}
