#![cfg_attr(test, feature(custom_test_frameworks, test))]
#![cfg_attr(test, test_runner(test_runner::uzu_harness))]

mod array;
// mod audio;
mod config;
mod encodable_block;
mod parameters;
pub mod speculators;
mod trie;
mod utils;

pub mod backends;
pub mod data_type;

pub mod engine;

pub use utils::{TOOLCHAIN_VERSION, VERSION};

// needed for tests to resolve `backend_uzu::` imports
#[cfg(test)]
extern crate self as backend_uzu;
#[cfg(test)]
pub mod tests;
