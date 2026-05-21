use thiserror::Error;

use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::ManualKernels,
    },
};

#[derive(Debug, Error)]
pub enum MatmulError<B: Backend> {
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported group size: {0}")]
    UnsupportedGroupSize(usize),
    #[error("Hadamard not supported for this kernel configuration")]
    UnsupportedHadamard,
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
}

pub enum MatmulArgumentC<'a, B: Backend> {
    None,
    /// Accumulate: [M, N]
    Accumulate,
    /// Bias: [N] (broadcasted across M/batch)
    Bias(&'a Allocation<B>),
}

pub enum MatmulWeights<'a, B: Backend> {
    FullPrecision {
        b: &'a Allocation<B>,
        b_offset: usize,
        b_leading_dimension: Option<u32>,
        b_transpose: bool,
        ab_scale: f32,
        c: MatmulArgumentC<'a, B>,
    },
    Quantized {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        zero_points_or_biases: &'a Allocation<B>,
        method: QuantizationMethod,
        mode: QuantizationMode,
        group_size: u32,
        hadamard_factors: Option<&'a Allocation<B>>,
    },
}

// D = ab_scale * (A @ op(B)) + C, where op(B) = B^T when b_transpose else B.
// For quantized weights, B is packed (4/8-bit) and dequantized internally.
pub struct MatmulArguments<'a, B: Backend> {
    /// A: [M, K]
    pub a: &'a Allocation<B>,
    pub a_offset: usize,
    /// B + weight-specific config.
    pub b: MatmulWeights<'a, B>,
    /// D: [M, N]
    pub d: &'a mut Allocation<B>,
    /// M dimension: usually batch/number of tokens.
    pub batch_dim: u32,
    /// K dimension: input/reduction dimension.
    pub input_dim: u32,
    /// N dimension: output dimension.
    pub output_dim: u32,
}

pub trait MatmulKernel: Sized {
    type Backend: Backend<Kernels: ManualKernels<MatmulKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Self::Backend>>;

    fn encode(
        &mut self,
        arguments: MatmulArguments<Self::Backend>,
        encoder: &mut Encoder<Self::Backend>,
    ) -> Result<(), MatmulError<Self::Backend>>;
}
