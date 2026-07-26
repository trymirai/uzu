#![allow(dead_code)] // TODO

#[cfg(not(backend = "cpu"))]
compile_error!("uzu tests require cpu backend");

pub extern crate test;

pub mod assert;
pub mod cold_pool;
pub mod helpers;
pub mod matmul;
pub mod proptest;
pub mod util;

#[path = "../../tests/unit/bench_cold_pool_test.rs"]
mod bench_cold_pool_test;
