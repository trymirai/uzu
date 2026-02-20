mod classifier;
mod classifier_context;
mod error;
mod result;

pub use classifier::Classifier;
pub use classifier_context::ClassifierContext;
pub use error::ClassifierError;
pub use result::{ClassificationOutput, ClassificationStats};

// Re-export encodables for convenience
pub use crate::encodable_block::{ClassifierLayer, ClassifierPredictionHead, Pooling};
#[cfg(feature = "tracing")]
pub use crate::forward_pass::traces::{ActivationTrace, LayerActivationTrace};
