pub mod harness;
pub mod quant;
pub mod shape;

#[cfg(metal_backend)]
pub use harness::run_metal;
pub use harness::{Case, cpu_reference, deterministic_input};
#[cfg(metal_backend)]
pub use quant::run_quant_metal;
pub use quant::{QuantBuffers, QuantInput, quant_arguments, run_quant_cpu};
pub use shape::{Shape, all_correctness_shapes};
