use crate::model::{ModelDescriptor, UzuModel};

pub trait UzuBackend {
    type Backend: UzuBackend;
    type Model: UzuModel;

    fn load_model(desc: &ModelDescriptor) -> Self::Model;

    fn identifier() -> String;

    fn version() -> String;
}
