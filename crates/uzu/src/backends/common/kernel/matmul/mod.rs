mod dispatch_descriptor;
mod full_precision;
pub mod gemm;
pub mod gemm_mpp;
pub mod gemm_mixed_types_simple;
pub mod gemv;
mod grid_size;
mod kernel;
mod matmul_arguments;

pub use dispatch_descriptor::MatmulDispatchDescriptor;
pub use full_precision::{FullPrecisionMatmulArguments, FullPrecisionMatmulKernel};
pub use grid_size::GridSize;
pub use kernel::MatmulKernel;
pub use matmul_arguments::MatmulArguments;
use thiserror::Error;

use super::Kernels;
use crate::backends::common::Backend;

pub trait MatmulKernels: Kernels {
    type FullPrecisionMatmulKernel: FullPrecisionMatmulKernel<Backend = Self::Backend>;
}

#[derive(Debug, Error)]
pub enum MatmulError<B: Backend> {
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(crate::DataType),
    #[error("Threadgroup dimension overflows u32: {0}")]
    ThreadgroupOverflow(usize),
    #[error("GEMV descriptor mismatch: apply_output_scale_and_accumulate=true but output_source=None")]
    GemvOutputSourceMismatch,
    #[error("GEMV descriptor requires bias buffer")]
    GemvMissingBias,
    #[error("GEMV stride overflows i32: {0}")]
    GemvStrideOverflow(i64),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
}
