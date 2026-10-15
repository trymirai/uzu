#![cfg_attr(test, feature(custom_test_frameworks))]
#![cfg_attr(test, test_runner(crate::tests::harness::uzu_harness))]
#![cfg_attr(target_family = "wasm", feature(wasi_ext))]

mod config;
mod data_type;
mod encodable_block;
mod safetensors;
mod speculators;
mod trie;
mod utils;

pub mod backends;

pub mod engine;

pub use config::token_codec::AnyTokenCodecConfig;
pub use utils::version::{TOOLCHAIN_VERSION, VERSION};

#[cfg(test)]
#[path = "../unit/common/mod.rs"]
pub mod tests;
