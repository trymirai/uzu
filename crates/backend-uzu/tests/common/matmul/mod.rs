pub mod harness;
pub mod shape;
pub mod variant;

pub use harness::{Case, cpu_reference, deterministic_input};
#[cfg(metal_backend)]
pub use harness::{alloc_bench_buffers, encode_iteration, run_metal};
pub use shape::{SHAPES_BENCH, Shape, all_correctness_shapes};
pub use variant::Variant;
