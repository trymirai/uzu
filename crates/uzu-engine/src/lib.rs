#![cfg_attr(test, feature(custom_test_frameworks, test))]
#![cfg_attr(test, test_runner(test_runner::uzu_harness))]

mod array;
mod config;
mod encodable_block;
mod parameters;
mod speculators;
mod trie;
mod utils;

pub mod backends;
pub mod bridge;
pub mod data_type;

pub mod engine;

pub use utils::version::{TOOLCHAIN_VERSION, VERSION};

#[cfg(test)]
#[path = "../unit/common/mod.rs"]
pub mod tests;
