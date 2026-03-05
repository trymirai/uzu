use super::{MatmulError, MatmulKernels};
use crate::{DataType, backends::common::Backend};

pub struct FullPrecisionMatmulArguments<'a, B: Backend> {
    pub a: &'a B::NativeBuffer,
    pub a_offset: usize,
    pub b: &'a B::NativeBuffer,
    pub output: &'a mut B::NativeBuffer,
    pub bias: Option<&'a B::NativeBuffer>,
    pub batch: usize,
    pub input_dim: usize,
    pub output_dim: usize,
}

pub trait FullPrecisionMatmulKernel: Sized {
    type Backend: Backend<Kernels: MatmulKernels<FullPrecisionMatmulKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Self::Backend>>;

    fn encode(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        encoder: &mut <Self::Backend as Backend>::ComputeEncoder,
        arguments: FullPrecisionMatmulArguments<Self::Backend>,
    );
}
