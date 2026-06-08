#![feature(custom_test_frameworks)]
#![test_runner(crate::bench_runner)]

fn bench_runner(benches: &[&dyn Fn()]) {
    #[cfg(target_os = "ios")]
    crate::common::path::ios_set_current_dir();
    crate::common::enable_benchmark_gpu_capture_if_requested();
    criterion::runner(benches);
}

use proc_macros::__internal_uzu_bench as uzu_bench;

#[macro_use]
#[path = "../../common/mod.rs"]
mod common;

#[path = "mod.rs"]
mod kernel;
