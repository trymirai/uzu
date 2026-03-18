mod matmul_arguments;

pub use matmul_arguments::MatmulArguments;
use thiserror::Error;

use super::Kernels;
use crate::{
    DataType,
    backends::common::{Backend, CommandBuffer},
};

pub trait MatmulKernels: Kernels {
    type MatmulKernel: MatmulKernel<Backend = Self::Backend>;
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
        arguments: MatmulArguments<Self::Backend>,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    );
}

#[derive(Debug, Error)]
pub enum MatmulError<B: Backend> {
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Threadgroup dimension overflows u32: {0}")]
    ThreadgroupOverflow(usize),
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
}
