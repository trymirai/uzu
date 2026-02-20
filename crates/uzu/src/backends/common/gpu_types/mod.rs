//! GPU types shared between Rust and shader languages (Metal, HLSL, Slang).
//!
//! These `#[repr(C)]` structs are the source of truth. The build system uses
//! cbindgen to generate C headers for Metal shaders.

pub mod attention;
pub mod kv_cache_update;
pub mod matmul;

pub use attention::*;
pub use kv_cache_update::*;
pub use matmul::*;
