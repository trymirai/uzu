// needed for tests to resolve `backend_uzu::` imports
#[cfg(test)]
extern crate self as backend_uzu;

mod array;
mod audio;
mod data_type;
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

// The following modules are private in production builds, but exposed publicly
// when the `tracing` feature is enabled so the trace-validation integration
// test (`tests/integration/tracer/trace_validator.rs`) can reach internal types.
macro_rules! tracing_visible_mod {
    ($($name:ident),* $(,)?) => {
        $(
            #[cfg(feature = "tracing")]
            pub mod $name;
            #[cfg(not(feature = "tracing"))]
            mod $name;
        )*
    };
}

tracing_visible_mod! {
    classifier,
    config,
    encodable_block,
    forward_pass,
    language_model,
    parameters,
}
