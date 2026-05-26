pub mod bench;
pub mod harness;
pub mod quant;
pub mod shape;

pub use bench::iter_encode_loop;
#[cfg(metal_backend)]
pub use harness::run_metal;
pub use harness::{Case, cpu_reference, deterministic_input};
#[cfg(metal_backend)]
pub use quant::run_quant_metal;
pub use quant::{
    CodebookQuantBuffers, CodebookQuantInput, QuantBuffers, QuantInput, codebook_quant_arguments, quant_arguments,
    run_quant_cpu,
};
pub use shape::{
    Shape, all_correctness_shapes, bench_fp_gemm_shapes, bench_quant_gemm_shapes, bench_quant_gemv_shapes,
    qwen3_layer_shapes,
};
