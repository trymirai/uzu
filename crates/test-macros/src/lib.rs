#![feature(test)]

extern crate test;

mod uzu_test;

pub use proc_macros::{uzu_bench, uzu_test};
pub use uzu_test::UzuTest;
