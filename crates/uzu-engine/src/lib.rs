#![cfg_attr(test, feature(custom_test_frameworks, test))]
#![cfg_attr(test, test_runner(crate::__uzu_test_harness::uzu_harness))]
#![cfg_attr(target_family = "wasm", feature(wasi_ext))]

#[cfg(test)]
#[path = "../unit/harness.rs"]
mod __uzu_test_harness;

mod array;
mod config;
mod encodable_block;
mod parameters;
mod speculators;
mod trie;
mod utils;

pub mod backends;
pub mod data_type;

pub mod engine;

pub use utils::version::{TOOLCHAIN_VERSION, VERSION};

#[cfg(test)]
#[path = "../unit/common/mod.rs"]
pub mod tests;
