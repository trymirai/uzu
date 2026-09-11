//! GPU types shared between Rust and shader languages (Metal, HLSL, Slang).
//!
//! These `#[repr(C)]` structs are the source of truth. The build system uses
//! cbindgen to generate C headers for Metal shaders.

pub mod activation_transform;
pub mod activation_type;
pub mod attention;
pub mod gated_act_mul;
pub mod gemm;
pub mod hadamard_order;
pub mod kv_cache_update;
pub mod matmul;
pub mod quantization;
pub mod quantization_method;
pub mod ring;
pub mod router_topk;
pub mod trie;
pub mod weaver;

pub use activation_transform::*;
pub use activation_type::*;
pub use attention::*;
pub use gated_act_mul::*;
pub use hadamard_order::*;
pub use kv_cache_update::*;
pub use matmul::*;
pub use quantization::*;
pub use quantization_method::*;
