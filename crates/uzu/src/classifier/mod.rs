mod classifier;
mod classifier_context;
mod error;
mod result;

pub use classifier::{Classifier, ClassifierTrait};
pub use classifier_context::ClassifierContext;
pub use error::ClassifierError;
pub use result::{ClassificationOutput, ClassificationStats};

#[cfg(feature = "tracing")]
pub use crate::forward_pass::traces::ActivationTrace;
