mod arguments;
mod d_ops;
mod error;
mod kernel;
mod matmul_a;
mod matmul_b;
pub mod routing;

pub use arguments::MatmulArguments;
pub(crate) use arguments::validate_matmul_storage;
pub use d_ops::MatmulDOps;
pub use error::MatmulError;
pub use kernel::MatmulKernel;
pub use matmul_a::MatmulA;
pub use matmul_b::MatmulB;
pub use routing::{A8ActivationPlan, ActivationFormat, ExpertInput, ExpertRoutes, MatmulRouting, MatmulShape};
