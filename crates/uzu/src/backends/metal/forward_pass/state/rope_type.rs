//! RoPE type enum.

use serde::{Deserialize, Serialize};

/// Type of RoPE (Rotary Position Embedding) buffers.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
)]
pub enum RopeType {
    Global,
    Local,
}

