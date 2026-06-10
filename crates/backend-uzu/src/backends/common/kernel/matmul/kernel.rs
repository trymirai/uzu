use crate::{
    backends::common::{
        AsBufferRangeRef, Backend, Buffer, Encoder, Kernels,
        kernel::matmul::{arguments::MatmulArguments, task::MatmulTask},
    },
    data_type::DataType,
};

pub trait MatmulKernel: Sized {
    type Backend: Backend<Kernels: Kernels<MatmulKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Self::Backend>>>(
        &mut self,
        arguments: MatmulArguments<Self::Backend, TB>,
        encoder: &mut Encoder<Self::Backend>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;

    /// Compiles every pipeline `encode` would select for `task` at each batch
    /// size, so the first real `encode` hits a warm cache. `task.m` is ignored;
    /// `precompile` sweeps `m` over `batch_sizes`.
    fn precompile(
        &mut self,
        context: &<Self::Backend as Backend>::Context,
        task: &MatmulTask,
        batch_sizes: &[u32],
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}
