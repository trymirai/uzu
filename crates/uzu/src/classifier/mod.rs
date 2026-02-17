mod classifier;
mod result;

pub use classifier::Classifier;
pub use result::{ClassificationOutput, ClassificationStats};

// Re-export context from metal backend
pub use crate::backends::metal::ClassifierContext;
// Re-export encodables from metal backend for convenience
pub use crate::backends::metal::encodable_block::ClassifierLayer;
pub use crate::encodable_block::{ClassifierPredictionHead, Pooling};
#[cfg(feature = "tracing")]
pub use crate::forward_pass::traces::{ActivationTrace, LayerActivationTrace};
