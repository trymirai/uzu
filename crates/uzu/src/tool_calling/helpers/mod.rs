//! Helper types for tool implementations (Rust-only, not FFI-compatible).

mod tool_implementation_callable;
mod tool_implementation_lambda;

pub use tool_implementation_callable::*;
pub(crate) use tool_implementation_lambda::*;
