#![feature(custom_test_frameworks)]
#![test_runner(crate::bench_runner)]

fn bench_runner(benches: &[&dyn Fn()]) {
    #[cfg(target_os = "ios")]
    crate::common::path::ios_set_current_dir();
    crate::common::enable_benchmark_gpu_capture_if_requested();
    criterion::runner(benches);
}

#[macro_use]
#[path = "../tests/common/mod.rs"]
mod common;
mod kernel;
mod language_model;
mod session;
