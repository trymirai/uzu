#![cfg_attr(test, feature(custom_test_frameworks, test))]
#![cfg_attr(test, test_runner(test_runner::uzu_harness))]

mod array;
mod config;
mod encodable_block;
mod parameters;
pub mod speculators;
mod staging;
mod trie;
mod utils;

pub mod backends;
pub mod bridge;
pub mod data_type;

pub mod engine;

pub use utils::version::{TOOLCHAIN_VERSION, VERSION};

// needed for tests to resolve `backend_uzu::` imports
#[cfg(test)]
extern crate self as backend_uzu;
#[cfg(test)]
pub mod tests;
