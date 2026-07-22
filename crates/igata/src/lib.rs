//! 鋳型 — the mold the kernel variants are cast in.
//!
//! Which kernels a build ships is decided here: the Rust `gpu_types` definitions are
//! parsed into axes and variant groups, the shader's `CONSTRAINT` expressions are
//! type-checked against them, and the surviving cross-product is enumerated into the
//! exact list of template-argument tuples to instantiate. The uzu build scripts wrap
//! this with the parts that touch the world — walking directories, driving clang,
//! writing files — so that everything above can be unit tested.

pub mod constraint_expr;
pub mod enum_paths;
pub mod gpu_types;
pub mod mangling;
pub mod variants;

#[cfg(test)]
mod tests;
