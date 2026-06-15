#![feature(custom_test_frameworks, test)]
#![test_runner(crate::tests::uzu_harness)]

#[macro_use]
#[path = "../tests/common/mod.rs"]
mod common;
#[path = "../src/tests/mod.rs"]
mod tests;

mod session;
