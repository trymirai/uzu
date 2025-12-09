//! HashMap identifier enum for forward pass.

/// Identifier for hashmaps used in the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashMapId {
    AttentionBias,
}
