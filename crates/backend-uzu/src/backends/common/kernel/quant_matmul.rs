use std::cell::RefCell;

use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::{QuantizationMethod, QuantizationMode},
        kernel::{ManualKernels, QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel},
    },
};

#[derive(Debug, thiserror::Error)]
pub enum QuantizedMatmulError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported group size: {0}")]
    UnsupportedGroupSize(usize),
    #[error("Hadamard not supported for this kernel configuration")]
    UnsupportedHadamard,
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizedMatmulConfiguration {
    pub data_type: DataType,
    pub group_size: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub mode: QuantizationMode,
    pub quantization_method: QuantizationMethod,
    pub use_hadamard: bool,
}

pub struct QuantizedMatmulArguments<'a, B: Backend> {
    pub a: &'a Allocation<B>,
    pub a_offset: usize,
    pub b: &'a Allocation<B>,
    pub scales: &'a Allocation<B>,
    pub zero_points_or_biases: &'a Allocation<B>,
    pub output: &'a mut Allocation<B>,
    pub hadamard_factors: Option<&'a Allocation<B>>,
    pub batch_dim: usize,
}

pub trait QuantizedGemmKernel: Sized {
    type Backend: Backend<Kernels: ManualKernels<QuantizedGemmKernel = Self>>;

    fn new(
        context: &<Self::Backend as Backend>::Context,
        configuration: QuantizedMatmulConfiguration,
    ) -> Result<Self, QuantizedMatmulError<Self::Backend>>;

    fn encode(
        &mut self,
        encoder: &mut Encoder<Self::Backend>,
        arguments: QuantizedMatmulArguments<Self::Backend>,
    );
}

pub struct QuantizedMatmulKernelEncodable<B: Backend> {
    matrix_vector: MatrixVectorKernel<B>,
    matrix_matrix: RefCell<<B::Kernels as ManualKernels>::QuantizedGemmKernel>,
    input_dim: usize,
    output_dim: usize,
    quantization_method: QuantizationMethod,
}

enum MatrixVectorKernel<B: Backend> {
    Qmv(<B::Kernels as Kernels>::QuantizedMatmulQmvKernel),
    QmvFast(<B::Kernels as Kernels>::QuantizedMatmulQmvFastKernel),
}

impl<B: Backend> QuantizedMatmulKernelEncodable<B> {
    pub fn new(
        context: &B::Context,
        configuration: QuantizedMatmulConfiguration,
    ) -> Result<Self, QuantizedMatmulError<B>> {
        if !matches!(configuration.data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(QuantizedMatmulError::UnsupportedDataType(configuration.data_type));
        }

        if !matches!(configuration.group_size, 32 | 64 | 128) {
            return Err(QuantizedMatmulError::UnsupportedGroupSize(configuration.group_size));
        }

        let bits = match configuration.mode {
            QuantizationMode::U4 => 4,
            QuantizationMode::I8 | QuantizationMode::U8 => 8,
        };
        let group_size = configuration.group_size as u32;

        let matrix_vector = if configuration.output_dim % 8 == 0 && configuration.input_dim % 512 == 0 {
            MatrixVectorKernel::QmvFast(
                <B::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
                    context,
                    configuration.data_type,
                    group_size,
                    bits,
                    configuration.quantization_method,
                    configuration.use_hadamard,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )
        } else {
            if configuration.use_hadamard {
                return Err(QuantizedMatmulError::UnsupportedHadamard);
            }

            MatrixVectorKernel::Qmv(
                <B::Kernels as Kernels>::QuantizedMatmulQmvKernel::new(
                    context,
                    configuration.data_type,
                    group_size,
                    bits,
                    configuration.quantization_method,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )
        };

        let matrix_matrix =
            <B::Kernels as ManualKernels>::QuantizedGemmKernel::new(context, configuration)?;

        Ok(Self {
            matrix_vector,
            matrix_matrix: RefCell::new(matrix_matrix),
            input_dim: configuration.input_dim,
            output_dim: configuration.output_dim,
            quantization_method: configuration.quantization_method,
        })
    }

    pub fn matrix_matrix(
        &self,
    ) -> std::cell::RefMut<'_, <B::Kernels as ManualKernels>::QuantizedGemmKernel> {
        self.matrix_matrix.borrow_mut()
    }

    pub fn encode(
        &self,
        encoder: &mut Encoder<B>,
        arguments: QuantizedMatmulArguments<B>,
    ) {
        if arguments.batch_dim >= 5 && self.output_dim > 1 {
            self.matrix_matrix.borrow_mut().encode(encoder, arguments);
            return;
        }

        let QuantizedMatmulArguments {
            a,
            a_offset,
            b,
            scales,
            zero_points_or_biases,
            output,
            hadamard_factors,
            batch_dim,
        } = arguments;

        let (zero_points, biases) = match self.quantization_method {
            QuantizationMethod::ScaleZeroPoint => (Some(zero_points_or_biases), None),
            QuantizationMethod::ScaleBias => (None, Some(zero_points_or_biases)),
        };

        macro_rules! encode_kernel {
            ($kernel:expr $(, $hadamard:expr)?) => {
                $kernel.encode(
                    b,
                    scales,
                    zero_points,
                    biases,
                    (a, a_offset),
                    output,
                    $($hadamard,)?
                    self.input_dim as u32,
                    self.output_dim as u32,
                    batch_dim as u32,
                    encoder,
                )
            };
        }

        match &self.matrix_vector {
            MatrixVectorKernel::Qmv(k) => encode_kernel!(k),
            MatrixVectorKernel::QmvFast(k) => encode_kernel!(k, hadamard_factors),
        }
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/backends/common/kernel/quant_matmul_test.rs"]
mod tests;
