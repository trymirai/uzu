mod dispatch_descriptor;
mod full_precision;
pub mod gemm;
pub mod gemv;
mod grid_size;
mod kernel;
mod matmul_arguments;
pub mod split_k;

pub use dispatch_descriptor::MatmulDispatchDescriptor;
pub use full_precision::{FullPrecisionMatmulArguments, FullPrecisionMatmulKernel};
pub use grid_size::GridSize;
pub use kernel::MatmulKernel;
pub use matmul_arguments::MatmulArguments;

use super::Kernels;

pub trait MatmulKernels: Kernels {
    type FullPrecisionMatmulKernel: FullPrecisionMatmulKernel<Backend = Self::Backend>;
}
