use crate::{
    backends::common::{Backend, BufferArg, Encoder, Kernels, kernel::matmul::arguments::MatmulArguments},
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

    fn supports_int8_symmetric_a(
        &self,
        context: &<Self::Backend as Backend>::Context,
        m: u32,
        n: u32,
        k: u32,
        group_size: u32,
    ) -> bool;

    fn encode<'a, 'b, 'd, TB: BufferArg<'b, Self::Backend>>(
        &mut self,
        arguments: MatmulArguments<'a, 'b, 'd, Self::Backend, TB>,
        encoder: &mut Encoder<Self::Backend>,
    ) -> Result<(), <Self::Backend as Backend>::Error>;
}
