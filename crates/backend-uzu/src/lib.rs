// needed for tests to resolve `backend_uzu::` imports
#[cfg(test)]
extern crate self as backend_uzu;

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
pub mod prelude;
pub mod session;

pub use array::{Array, ArrayContextExt};
#[cfg(metal_backend)]
pub use audio::{NanoCodecFsqRuntime, NanoCodecFsqRuntimeConfig};
pub use backends::common::{AllocationAccessError, allocation_copy_from_slice, allocation_to_vec};
pub use data_type::{ArrayElement, DataType};
pub use language_model::gumbel::{gumbel_float, revidx};
pub use parameters::{ParameterLoader, read_safetensors_metadata};
pub use utils::{TOOLCHAIN_VERSION, VERSION};

#[cfg(feature = "tracing")]
pub mod _private {
    pub use crate::{
        classifier::Classifier,
        config::{ModelConfig, ModelMetadata, ModelType},
        encodable_block::{DecoderDecodeInput, Sampling},
        forward_pass::{
            cache_layers::CacheLayers, kv_cache_layer::KVCacheLayer, token_inputs::TokenInputs, traces::ActivationTrace,
        },
        language_model::{
            language_model_generator_context::LanguageModelGeneratorContext,
            sampler::{ArgmaxSampler, LogitsSampler},
        },
        parameters::ParameterTree,
    };
}
