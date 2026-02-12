use super::MatmulKernels;
use crate::{DataType, backends::common::Backend, config::QuantizationMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedMatmulType {
    ZeroPoint,
    Mlx,
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizedMatmulConfiguration {
    pub data_type: DataType,
    pub group_size: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub mode: QuantizationMode,
    pub quantization_type: QuantizedMatmulType,
    pub weights_transposed: bool,
}

pub struct QuantizedMatmulArguments<'a, B: Backend> {
    pub a_buffer: &'a B::NativeBuffer,
    pub a_offset: usize,
    pub b_buffer: &'a B::NativeBuffer,
    pub scales_buffer: &'a B::NativeBuffer,
    pub zero_points_or_biases_buffer: &'a B::NativeBuffer,
    pub output_buffer: &'a B::NativeBuffer,
    pub batch: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub quantization_type: QuantizedMatmulType,
}

pub trait QuantizedMatmulKernel: Sized {
    type Backend: Backend<Kernels: MatmulKernels<QuantizedMatmulKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        configuration: QuantizedMatmulConfiguration,
    ) -> Result<Self, <Self::Backend as Backend>::Error>;

    fn encode(
        &self,
        encoder: &<Self::Backend as Backend>::ComputeEncoder,
        arguments: QuantizedMatmulArguments<Self::Backend>,
    );
}
