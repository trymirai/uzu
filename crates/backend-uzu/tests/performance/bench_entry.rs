#![feature(custom_test_frameworks)]
#![test_runner(crate::bench_runner)]

fn bench_runner(benches: &[&dyn Fn()]) {
    criterion::runner(benches);
}

use proc_macros::__internal_uzu_bench as uzu_bench;

#[macro_use]
#[path = "../common/mod.rs"]
mod common;
mod session;
