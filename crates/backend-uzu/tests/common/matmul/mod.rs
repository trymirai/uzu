pub mod harness;
pub mod shape;

#[cfg(metal_backend)]
pub use harness::run_metal;
pub use harness::{Case, cpu_reference, deterministic_input};
pub use shape::{Shape, all_correctness_shapes};
