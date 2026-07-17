//! GPU types shared between Rust and shader languages (Metal, HLSL, Slang).
//!
//! These Rust declarations, including `#[repr(C)]` types and public constants,
//! are the source of truth for generated shader headers.

pub mod activation_prepare;
pub mod activation_type;
pub mod argmax;
pub mod attention;
pub mod gemm;
pub mod hadamard_order;
pub mod kv_cache_update;
pub mod matmul;
pub mod quantization;
pub mod quantization_method;
pub mod ring;
pub mod trie;

pub use activation_prepare::*;
pub use activation_type::*;
pub use argmax::*;
pub use attention::*;
pub use hadamard_order::*;
pub use kv_cache_update::*;
pub use matmul::*;
pub use quantization::*;
pub use quantization_method::*;
