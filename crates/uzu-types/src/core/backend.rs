use crate::{core::model::InferenceModel, model::ModelSource};

/// A backend that loads models
pub trait InferenceBackend {
    fn load_model(desc: &ModelSource) -> Box<dyn InferenceModel>;

    fn identifier() -> String;

    fn version() -> String;
}
