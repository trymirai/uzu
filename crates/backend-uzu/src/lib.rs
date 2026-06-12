#![cfg_attr(test, feature(custom_test_frameworks, test))]
#![cfg_attr(test, test_runner(crate::harness::uzu_harness))]

// needed for tests to resolve `backend_uzu::` imports
#[cfg(test)]
extern crate self as backend_uzu;

#[cfg(test)]
#[macro_use]
#[path = "../tests/common/mod.rs"]
mod common;

#[cfg(test)]
#[path = "../unit/bench_cold_pool_test.rs"]
mod bench_cold_pool_test;

pub mod array;
mod audio;
mod classifier;
mod config;
pub mod data_type;
mod encodable_block;
mod forward_pass;
mod language_model;
mod parameters;
pub mod speculators;
mod trie;
mod utils;

pub mod backends;
pub mod inference;
pub mod session;

pub use utils::{TOOLCHAIN_VERSION, VERSION};

#[doc(hidden)]
pub mod _benchmarks {
    pub use crate::{
        config::model::language_model::LanguageModelConfig,
        language_model::{LanguageModelGenerator, language_model_generator::RunModelResult},
        trie::{TrieCreationConfig, TrieNode},
    };
}

#[cfg(test)]
#[path = "../tests/harness/mod.rs"]
mod harness;
