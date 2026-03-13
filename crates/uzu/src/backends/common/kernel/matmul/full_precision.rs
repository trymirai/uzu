use super::{MatmulError, MatmulKernels};
use crate::{
    DataType,
    backends::common::{Backend, CommandBuffer},
};

pub struct FullPrecisionMatmulArguments<'a, B: Backend> {
    pub a: &'a B::Buffer,
    pub a_offset: usize,
    pub b: &'a B::Buffer,
    pub output: &'a mut B::Buffer,
    pub bias: Option<&'a B::Buffer>,
    pub batch: usize,
    pub input_dim: usize,
    pub output_dim: usize,
}

pub trait MatmulKernel: Sized {
    type Backend: Backend<Kernels: MatmulKernels<MatmulKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Self::Backend>>;

    fn encode(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
        arguments: FullPrecisionMatmulArguments<Self::Backend>,
    );
}
