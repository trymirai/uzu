mod full_precision;
pub mod matmul_arguments;

pub use full_precision::{FullPrecisionMatmulArguments, MatmulKernel};
pub use matmul_arguments::MatmulArguments;
use thiserror::Error;

use super::Kernels;
use crate::backends::common::Backend;

pub trait MatmulKernels: Kernels {
    type MatmulKernel: MatmulKernel<Backend = Self::Backend>;
}

#[derive(Debug, Error)]
pub enum MatmulError<B: Backend> {
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(crate::DataType),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
}
