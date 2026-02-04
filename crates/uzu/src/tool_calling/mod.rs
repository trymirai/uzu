//! Tool calling infrastructure for LLM function calling.
//!
//! This module provides FFI-compatible types and a registry for implementing
//! tool/function calling with large language models.

mod helpers;
mod registry;
pub mod types;

pub use helpers::*;
pub use registry::*;
pub use types::*;

// Re-export the macro
pub use uzu_macros::tool;
