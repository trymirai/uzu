//! GPU types shared between Rust and shader languages (Metal, HLSL, Slang).
//!
//! These `#[repr(C)]` structs are the source of truth. The build system uses
//! cbindgen to generate C headers for Metal shaders.

pub mod activation_type;
pub mod argmax;
pub mod attention;
pub mod kv_cache_update;
pub mod matmul;
pub mod quantization;
pub mod quantized_format;
pub mod ring;
pub mod trie;
pub mod unified_gemm;

pub use activation_type::*;
pub use argmax::*;
pub use attention::*;
pub use kv_cache_update::*;
pub use matmul::*;
pub use quantization::*;
pub use quantized_format::*;
pub use unified_gemm::*;
