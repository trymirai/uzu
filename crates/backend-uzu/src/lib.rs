#![cfg_attr(test, feature(custom_test_frameworks, test))]
#![cfg_attr(test, test_runner(crate::tests::uzu_harness))]

mod array;
mod audio;
mod classifier;
mod config;
mod data_type;
mod encodable_block;
mod forward_pass;
mod language_model;
mod parameters;
mod speculators;
mod trie;
mod utils;

pub mod backends;
pub mod inference;
pub mod session;

pub use utils::{TOOLCHAIN_VERSION, VERSION};

#[cfg(test)]
pub mod tests;
