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
#[cfg(metal_backend)]
pub use crate::backends::metal::MetalContext;
#[cfg(all(feature = "audio-runtime", metal_backend))]
pub use crate::session::TtsSession;
pub use crate::{
    VERSION,
    audio::{AudioCodecRuntime, AudioError, AudioPcmBatch, AudioResult, AudioTokenGrid},
    classifier::{ClassificationOutput, ClassificationStats},
    parameters::{HeaderLoadingError, ParameterLeaf, ParameterLoaderError, ParameterTree},
    session::{
        ChatSession, ClassificationSession,
        config::{DecodingConfig, RunConfig, SpeculatorConfig},
        parameter::{ContextLength, ContextMode, PrefillStepSize, SamplingMethod, SamplingPolicy, SamplingSeed},
        types::{
            Error, FinishReason, Input, Message, Output, ParsedText, Role, RunStats, Stats, StepStats, Text, TotalStats,
        },
    },
    speculators::{empty_speculator::EmptySpeculator, ngram_speculator::NGramSpeculator, speculator::Speculator},
};
