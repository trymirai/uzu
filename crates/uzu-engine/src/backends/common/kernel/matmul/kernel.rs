use std::error::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, Kernels,
        kernel::{
            ActivationQuantization,
            matmul::{
                arguments::MatmulArguments,
                routing::{ActivationFormat, MatmulShape},
            },
        },
    },
    data_type::DataType,
};

pub trait MatmulKernel: Sized + Send + Sync {
    type Backend: Backend<Kernels: Kernels<MatmulKernel = Self>>;
    type Error: Error + 'static;
    type WeightsLayout: Copy + Default + PartialEq + Send + Sync;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, Self::Error>;

    fn encode(
        &mut self,
        arguments: MatmulArguments<
            '_,
            Self::Backend,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
            impl BufferMut<Backend = Self::Backend>,
            impl BufferRef<Backend = Self::Backend>,
        >,
        command_buffer: &mut <<Self::Backend as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), Self::Error>;

    fn select_activation_quantization(
        &self,
        _candidate: &MatmulShape,
        _context: &<Self::Backend as Backend>::Context,
    ) -> Option<ActivationQuantization> {
        None
    }

    fn select_activation_format(
        &self,
        _bf16_shape: &MatmulShape,
        _weights_layout: Self::WeightsLayout,
        _context: &<Self::Backend as Backend>::Context,
    ) -> ActivationFormat {
        ActivationFormat::Bf16
    }

    fn optimize_weights_layout(
        &self,
        a8_shape: &MatmulShape,
        _weights: &mut <Self::Backend as Backend>::GlobalGpuBuffer,
        _context: &<Self::Backend as Backend>::Context,
    ) -> Result<(Self::WeightsLayout, bool), <Self::Backend as Backend>::Error> {
        Ok((Self::WeightsLayout::default(), a8_shape.signed_codes))
    }
}
