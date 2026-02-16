mod full_precision;
mod quantized;

pub use full_precision::{FullPrecisionMatmulArguments, FullPrecisionMatmulKernel};
pub use quantized::{
    QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulKernel, QuantizedMatmulType,
};

use super::Kernels;

pub trait MatmulKernels: Kernels {
    type FullPrecisionMatmulKernel: FullPrecisionMatmulKernel<Backend = Self::Backend>;
    type QuantizedMatmulKernel: QuantizedMatmulKernel<Backend = Self::Backend>;
}
