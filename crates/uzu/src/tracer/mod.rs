//! Unified trace validation for all model types.
//!
//! This module provides `TraceValidator` which can validate activation traces
//! for any model type (LLM or classifier). The model type is automatically
//! detected from the config.

mod trace_validator;

pub use trace_validator::{
    ArrayTransform, TraceValidator, TracerValidationMetrics,
    TracerValidationResult, TracerValidationResults,
};
