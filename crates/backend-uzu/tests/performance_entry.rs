#![feature(custom_test_frameworks, test)]
#![test_runner(crate::harness::uzu_harness)]

#[path = "harness/mod.rs"]
mod harness;

mod common;
mod performance;
