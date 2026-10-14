#![cfg_attr(test, feature(custom_test_frameworks))]
#![cfg_attr(test, test_runner(crate::tests::harness::uzu_harness))]
#![cfg_attr(target_os = "wasi", feature(wasi_ext))]

mod array;
mod config;
mod encodable_block;
mod parameters;
pub use parameters::{ParameterBufferInitializer, ParameterLoader, ParameterLoaderError, ParameterReadSource};
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
