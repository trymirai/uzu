mod arguments;
mod kernel;
mod pipeline;
#[allow(dead_code, clippy::unnecessary_operation, clippy::identity_op)]
mod shared_types;
mod transpose;

pub use arguments::MatmulArguments;
pub use kernel::MatmulKernel;
