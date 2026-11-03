//! Apple Neural Engine (ANE) runtime for prefill offload.
//!
//! The ANE is reached through the private `AppleNeuralEngine.framework` (the service behind
//! Core ML): programs are MIL text compiled by the ANE service, and their inputs and outputs are
//! IOSurfaces that Metal aliases zero-copy, so the GPU and the ANE exchange data without copies.
//!
//! - [`AneRuntime`]: the framework's classes and client, resolved and checked once.
//! - [`AneSurface`]: an IOSurface in the ANE's channel-major layout, aliased by a Metal buffer.
//! - [`AneProgram`]: a compiled, loaded multi-function program; synchronous evaluation or
//!   asynchronous evaluation ordered by a Metal shared event.
//! - [`mil`]: the W8A8 FFN program text and its constant blob.
//! - [`fake`]: the CPU reference of the FFN program, the test oracle.
//!
//! Portions are adapted from Splash (<https://github.com/incoai/splash>, commit 031a25a),
//! Apache License 2.0: `runtime/ane/Program.mm` (client calls, IOSurface layout, constant blob)
//! and `runtime/ops/AneFfn.cpp` (the W8A8 FFN MIL program). Not affiliated with Splash.

pub mod fake;
pub mod ffn;
pub mod mil;
pub mod mlp;
mod objc;
mod program;
mod runtime;
mod surface;

pub use program::{AneBinding, AneProcedure, AneProgram};
pub use runtime::AneRuntime;
pub use surface::{AneElement, AneSurface};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AneError {
    #[error("ANE unavailable: {0}")]
    Unavailable(String),
    #[error("ANE {step} failed: {message}")]
    Call {
        step: &'static str,
        message: String,
    },
    #[error("ANE program description is malformed: {0}")]
    Description(String),
    #[error("ANE program has no procedure {0}")]
    UnknownProcedure(String),
    #[error("ANE binding lacks input {0}")]
    MissingInput(String),
    #[error("IOSurface creation failed: {0}")]
    Surface(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
#[path = "../../../../unit/backends/metal/ane/mod.rs"]
mod tests;
