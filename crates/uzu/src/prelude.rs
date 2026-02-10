//! Prelude module containing commonly used types from the uzu crate.
//!
//! This module can be imported with `use uzu::prelude::*;` to bring the most
//! frequently used types into scope.

// Constants & Utilities
// Session Core
// Session Config
pub use crate::session::config::{DecodingConfig, RunConfig, SpeculatorConfig};
// Session Parameters
pub use crate::session::parameter::{
    ContextLength, ContextMode, PrefillStepSize, SamplingMethod, SamplingPolicy, SamplingSeed,
};
// Session Types
pub use crate::session::types::{
    Error, FinishReason, Input, Message, Output, ParsedText, Role, RunStats, Stats, StepStats, Text, TotalStats,
};
// Speculators
pub use crate::speculators::speculator::Speculator;
pub use crate::{
    VERSION,
    session::{ChatSession, ClassificationSession},
    speculators::empty_speculator::EmptySpeculator,
    storage_path,
};
