#[macro_use]
#[path = "../common/mod.rs"]
mod common;

#[cfg(metal_backend)]
mod matmul;
mod model_loading_perf_test;
