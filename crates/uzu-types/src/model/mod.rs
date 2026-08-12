use crate::backend::UzuBackend;

mod descriptor;

pub use descriptor::ModelDescriptor;

pub trait UzuModel {
    type Backend: UzuBackend;

    // TODO
}
