use thiserror::Error;

// Classifier context errors
#[derive(Debug, Error)]
pub enum ClassifierError {
    #[error("Missing required config field: {0}")]
    MissingConfigField(String),
    #[error("Model requires attention layers but found non-attention mixer")]
    NonAttentionMixer,
    #[error("Weight subtree not found: {0}")]
    WeightSubtreeNotFound(String),
    #[error("Kernel creation failed: {0}")]
    KernelCreationFailed(String),
    #[error("{0}")]
    Custom(String),
}
