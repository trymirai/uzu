mod classifier;
mod result;

pub use classifier::Classifier;
pub use result::{ClassificationOutput, ClassificationStats};

// Re-export context from metal backend
pub use crate::backends::metal::ClassifierContext;
// Re-export encodables from metal backend for convenience
pub use crate::backends::metal::encodable_block::{
    ClassifierLayer, ClassifierPredictionHead, Pooling,
};
#[cfg(feature = "tracing")]
pub use crate::backends::metal::forward_pass::traces::{
    ActivationTrace, LayerActivationTrace,
};
