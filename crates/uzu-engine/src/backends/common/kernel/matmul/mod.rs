mod arguments;
mod d_ops;
mod error;
pub mod group_major_metadata;
pub mod interleaved_w4;
mod kernel;
mod matmul_a;
mod matmul_b;
mod parallel_rows;
pub mod routing;

pub use arguments::MatmulArguments;
pub use d_ops::MatmulDOps;
pub use error::MatmulError;
pub use kernel::MatmulKernel;
pub use matmul_a::MatmulA;
pub use matmul_b::{MatmulB, MetadataLayout};
pub use routing::{A8ActivationPlan, ActivationFormat, MatmulShape};
