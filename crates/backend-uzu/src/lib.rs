// needed for tests to resolve `backend_uzu::` imports
extern crate self as backend_uzu;

mod array;
mod audio;
mod classifier;
mod config;
mod data_type;
mod encodable_block;
mod forward_pass;
pub mod inference;
mod language_model;
mod parameters;
mod speculators;
#[cfg(feature = "tracing")]
mod tracer;
mod trie;
mod utils;

pub mod backends;
pub mod prelude;
pub mod session;

pub use array::{Array, ArrayContextExt};
#[cfg(metal_backend)]
pub use audio::{NanoCodecFsqRuntime, NanoCodecFsqRuntimeConfig};
pub use config::ConfigDataType;
pub use data_type::{ArrayElement, DataType};
pub use language_model::gumbel::{gumbel_float, revidx};
pub use parameters::{ParameterLoader, read_safetensors_metadata};
#[cfg(feature = "tracing")]
pub use tracer::TraceValidator;
pub use utils::{TOOLCHAIN_VERSION, VERSION};
