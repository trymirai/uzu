use thiserror::Error;

use crate::{
    DataType,
    backends::common::{Backend, Encoder, kernel::ManualKernels},
};

#[derive(Debug, Error)]
pub enum MatmulError<B: Backend> {
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
}

#[derive(Debug)]
pub enum MatmulArgumentC<'a, B: Backend> {
    None,
    /// Accumulate: [M, N]
    Accumulate,
    /// Bias: [N] (broadcasted across M/batch)
    Bias(&'a B::DenseBuffer),
}

// D = ab_scale * (A @ B.T) + C
#[derive(Debug)]
pub struct MatmulArguments<'a, B: Backend> {
    /// A: [M, K]
    pub a: &'a B::DenseBuffer,
    pub a_offset: u64,
    /// B: [N, K]
    pub b: &'a B::DenseBuffer,
    /// AB scale: also known as alpha
    pub ab_scale: f32,
    /// C: behavior depends on enum variant
    pub c: MatmulArgumentC<'a, B>,
    /// D: [M, N]
    pub d: &'a mut B::DenseBuffer,
    /// M dimension: usually batch/number of tokens (rows of A, rows of C)
    pub batch_dim: u32,
    /// K dimension: usually input_dim/reduction dimension (cols of A, rows of B)
    pub input_dim: u32,
    /// N dimension: usually output_dim (cols of B, cols of C)
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
        context: &<Self::Backend as Backend>::Context,
        arguments: MatmulArguments<Self::Backend>,
        encoder: &mut Encoder<Self::Backend>,
    );
}
