mod error;
mod kernel;
mod specialization;

pub use error::GemmSpecializationError;
pub use kernel::{GemmDispatchPath, GemmKernel};
