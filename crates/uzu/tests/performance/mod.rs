#[macro_use]
#[path = "../common/mod.rs"]
mod common;

mod encodable_block;
#[cfg(metal_backend)]
mod matmul;
mod model_loading_perf_test;
