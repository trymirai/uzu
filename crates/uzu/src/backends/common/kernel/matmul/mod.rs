mod full_precision;

pub use full_precision::{FullPrecisionMatmulArguments, FullPrecisionMatmulKernel};

use super::Kernels;

pub trait MatmulKernels: Kernels {
    type FullPrecisionMatmulKernel: FullPrecisionMatmulKernel<Backend = Self::Backend>;
}
