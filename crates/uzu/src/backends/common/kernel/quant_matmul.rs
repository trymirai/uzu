use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::QuantizationMode,
        kernel::{
            QuantizedMatmulQmmTransposed64x64Kernel, QuantizedMatmulQmmTransposedKernel,
            QuantizedMatmulQmmTransposedWideKernel, QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel,
        },
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
}

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
}

pub struct QuantizedMatmulArguments<'a, 'input, 'output, B: Backend> {
    pub a: &'input Allocation<B>,
    pub b: &'a Allocation<B>,
    pub scales: &'a Allocation<B>,
    pub zero_points_or_biases: &'a Allocation<B>,
    pub output: &'output mut Allocation<B>,
    pub batch_dim: usize,
}

pub struct QuantizedMatmulKernelEncodable<B: Backend> {
    matrix_vector: MatrixVectorKernel<B>,
    matrix_matrix: MatrixMatrixKernel<B>,
    input_dim: usize,
    output_dim: usize,
    quantization_type: QuantizedMatmulType,
}

enum MatrixVectorKernel<B: Backend> {
    Qmv(<B::Kernels as Kernels>::QuantizedMatmulQmvKernel),
    QmvFast(<B::Kernels as Kernels>::QuantizedMatmulQmvFastKernel),
}

enum MatrixMatrixKernel<B: Backend> {
    QmmTransposed(<B::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel),
    QmmTransposed64x64(<B::Kernels as Kernels>::QuantizedMatmulQmmTransposed64x64Kernel),
    QmmTransposedWide(<B::Kernels as Kernels>::QuantizedMatmulQmmTransposedWideKernel),
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
            QuantizationMode::UINT4 => 4,
            QuantizationMode::INT8 | QuantizationMode::UINT8 => 8,
        };
        let group_size = configuration.group_size as u32;
        let use_mlx_quant = matches!(configuration.quantization_type, QuantizedMatmulType::Mlx);
        let use_zero_points = !use_mlx_quant;

        // Matrix-vector
        let matrix_vector = if configuration.output_dim % 8 == 0 && configuration.input_dim % 512 == 0 {
            MatrixVectorKernel::QmvFast(
                <B::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
                    context,
                    configuration.data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )
        } else {
            MatrixVectorKernel::Qmv(
                <B::Kernels as Kernels>::QuantizedMatmulQmvKernel::new(
                    context,
                    configuration.data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )
        };

        // Matrix-matrix
        let aligned_n_64 = configuration.output_dim % 64 == 0;
        let is_bf16 = configuration.data_type == DataType::BF16;

        let matrix_matrix = if aligned_n_64 && is_bf16 && matches!(configuration.group_size, 64 | 128) {
            MatrixMatrixKernel::QmmTransposed64x64(
                <B::Kernels as Kernels>::QuantizedMatmulQmmTransposed64x64Kernel::new(
                    context,
                    configuration.data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )
        } else if aligned_n_64 && is_bf16 {
            MatrixMatrixKernel::QmmTransposedWide(
                <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedWideKernel::new(
                    context,
                    configuration.data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )
        } else {
            let aligned_n = configuration.output_dim % 32 == 0;
            MatrixMatrixKernel::QmmTransposed(
                <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel::new(
                    context,
                    configuration.data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
                    aligned_n,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )
        };

        Ok(Self {
            matrix_vector,
            matrix_matrix,
            input_dim: configuration.input_dim,
            output_dim: configuration.output_dim,
            quantization_type: configuration.quantization_type,
        })
    }

    pub fn encode<'a, 'input, 'output>(
        &self,
        encoder: &mut Encoder<B>,
        arguments: QuantizedMatmulArguments<'a, 'input, 'output, B>,
    ) -> Result<(), QuantizedMatmulError<B>> {
        let QuantizedMatmulArguments {
            a,
            b,
            scales,
            zero_points_or_biases,
            output,
            batch_dim,
        } = arguments;

        let (zero_points, biases) = match self.quantization_type {
            QuantizedMatmulType::ZeroPoint => (Some(zero_points_or_biases), None),
            QuantizedMatmulType::Mlx => (None, Some(zero_points_or_biases)),
        };

        macro_rules! encode_kernel {
            ($kernel:expr) => {
                $kernel.encode(
                    b,
                    scales,
                    zero_points,
                    biases,
                    a,
                    output,
                    self.input_dim as u32,
                    self.output_dim as u32,
                    batch_dim as u32,
                    encoder,
                )
            };
        }

        if batch_dim < 32 || self.output_dim == 1 {
            match &self.matrix_vector {
                MatrixVectorKernel::Qmv(k) => encode_kernel!(k),
                MatrixVectorKernel::QmvFast(k) => encode_kernel!(k),
            }
        } else {
            match &self.matrix_matrix {
                MatrixMatrixKernel::QmmTransposed(k) => encode_kernel!(k),
                MatrixMatrixKernel::QmmTransposed64x64(k) => encode_kernel!(k),
                MatrixMatrixKernel::QmmTransposedWide(k) => encode_kernel!(k),
            }
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/backends/common/kernel/quant_matmul_test.rs"]
mod tests;
