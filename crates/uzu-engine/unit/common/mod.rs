#[cfg(not(backend = "cpu"))]
compile_error!("uzu tests require cpu backend");

pub extern crate test;

pub mod assert;
pub mod cold_pool;
pub mod env_vars;
pub mod helpers;
pub mod matmul;
pub mod metrics;
pub mod path;
pub mod perf;
pub mod proptest;
pub mod util;

#[path = "../bench_cold_pool_test.rs"]
mod bench_cold_pool_test;
#[path = "../session/model_loading_bench.rs"]
mod model_loading_bench;
