#![cfg_attr(test, feature(custom_test_frameworks))]
#![cfg_attr(test, test_runner(crate::tests::harness::uzu_harness))]

mod config;
mod data_type;
mod encodable_block;
mod safetensors;
mod speculators;
mod trie;
mod utils;
pub mod verbose;

pub mod audio;
pub mod backends;
pub mod d1_omni;
pub mod engine;
#[cfg(backend = "metal")]
pub mod haru;
pub mod omni;
pub mod vision;

pub use utils::version::{TOOLCHAIN_VERSION, VERSION};

#[cfg(test)]
#[path = "../unit/common/mod.rs"]
pub mod tests;
