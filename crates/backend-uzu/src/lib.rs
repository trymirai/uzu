// needed for tests to resolve `backend_uzu::` imports
#[cfg(test)]
extern crate self as backend_uzu;

#[cfg(test)]
#[macro_use]
#[path = "../tests/common/mod.rs"]
mod common;

pub mod array;
mod audio;
mod classifier;
mod config;
pub mod data_type;
mod encodable_block;
mod forward_pass;
mod language_model;
pub mod parameters;
mod speculators;
mod trie;
mod utils;

pub mod backends;
pub mod inference;
pub mod prelude;
pub mod session;

#[cfg(metal_backend)]
pub use audio::{NanoCodecFsqRuntime, NanoCodecFsqRuntimeConfig};
pub use language_model::gumbel::{gumbel_float, revidx};
pub use utils::{TOOLCHAIN_VERSION, VERSION};

#[doc(hidden)]
pub mod _benchmarks {
    pub use crate::{
        config::model::language_model::LanguageModelConfig,
        language_model::{LanguageModelGenerator, language_model_generator::RunModelResult},
        trie::{TrieCreationConfig, TrieNode},
    };
}

#[cfg(feature = "tracing")]
pub mod _private {
    pub use crate::{
        classifier::Classifier,
        config::model::AnyModelConfig,
        encodable_block::{DecoderDecodeInput, Sampling},
        forward_pass::{
            cache_layers::CacheLayers, kv_cache_layer::KVCacheLayer, token_inputs::TokenInputs, traces::ActivationTrace,
        },
        language_model::language_model_generator_context::LanguageModelGeneratorContext,
    };
}
