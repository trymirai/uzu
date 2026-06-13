#![feature(custom_test_frameworks, test)]
#![test_runner(crate::tests::uzu_harness)]

#[path = "../src/tests/mod.rs"]
mod tests;

mod performance;
