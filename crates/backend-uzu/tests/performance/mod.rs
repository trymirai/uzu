#[macro_use]
#[path = "../common/mod.rs"]
mod common;

mod audio_kernel_perf_test;
mod encodable_block;
#[cfg(metal_backend)]
mod matmul;
mod model_loading_perf_test;
