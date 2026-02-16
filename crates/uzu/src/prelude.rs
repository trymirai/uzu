//! Prelude module containing commonly used types from the uzu crate.
//!
//! This module can be imported with `use uzu::prelude::*;` to bring the most
//! frequently used types into scope.

// Constants & Utilities
// Session Core
// Session Config
// Session Parameters
// Session Types
// Speculators
pub use crate::{
    VERSION,
    session::{
        ChatSession, ClassificationSession,
        config::{DecodingConfig, RunConfig, SpeculatorConfig},
        parameter::{ContextLength, ContextMode, PrefillStepSize, SamplingMethod, SamplingPolicy, SamplingSeed},
        types::{
            Error, FinishReason, Input, Message, Output, ParsedText, Role, RunStats, Stats, StepStats, Text, TotalStats,
        },
    },
    speculators::{empty_speculator::EmptySpeculator, speculator::Speculator},
    storage_path,
};
