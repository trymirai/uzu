//! AMDGPU kernels: the MSL sources of the Metal backend compiled with clang as C++ for OpenCL.
//!
//! The DSL front end (AST annotations, variants, constraints) is shared with the Metal compiler and
//! included from `build/metal` as is; this module owns the clang toolchain, the `__kernel` wrappers
//! and the HIP-side Rust bindings.

#[path = "../metal/ast.rs"]
#[allow(dead_code)]
mod ast;
#[path = "../metal/enum_path_rewrite.rs"]
#[allow(dead_code)]
mod enum_path_rewrite;
#[path = "../metal/variant_combinations.rs"]
mod variant_combinations;

mod bindgen;
mod compiler;
mod native;
mod sharding;
mod toolchain;
mod wrapper;

pub use compiler::AmdgpuCompiler;
