use crate::{
    DataType,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::QuantizationMode,
        kernel::{QuantizedMatmulQmmTransposedKernel, QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel},
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
    pub use_hadamard: bool,
}

pub struct QuantizedMatmulArguments<'a, B: Backend> {
    pub a: &'a Allocation<B>,
    pub b: &'a Allocation<B>,
    pub scales: &'a Allocation<B>,
    pub zero_points_or_biases: &'a Allocation<B>,
    pub output: &'a mut Allocation<B>,
    pub hadamard_factors: Option<&'a Allocation<B>>,
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

type QmmKernel<B> = <<B as Backend>::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel;

enum MatrixMatrixKernel<B: Backend> {
    Big(QmmKernel<B>),
    BigAndSmall {
        big: QmmKernel<B>,
        small: QmmKernel<B>,
    },
}

impl<B: Backend> MatrixMatrixKernel<B> {
    fn pick(
        &self,
        batch_dim: usize,
    ) -> Option<&QmmKernel<B>> {
        match self {
            Self::Big(big) => (batch_dim >= 32).then_some(big),
            Self::BigAndSmall {
                big,
                small,
            } => Some(if batch_dim < 48 {
                small
            } else {
                big
            }),
        }
    }
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

        let matrix_vector = if configuration.output_dim % 8 == 0 && configuration.input_dim % 512 == 0 {
            MatrixVectorKernel::QmvFast(
                <B::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
                    context,
                    configuration.data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
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
                    use_zero_points,
                    use_mlx_quant,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )
        };

        let aligned_n_64 = configuration.output_dim % 64 == 0;
        let aligned_n_32 = configuration.output_dim % 32 == 0;
        let is_bf16 = configuration.data_type == DataType::BF16;
        let can_use_64_tile = aligned_n_64 && is_bf16;

        let (bm, bk, bn, aligned_n) = if can_use_64_tile && matches!(configuration.group_size, 64 | 128) {
            (64u32, 64u32, 64u32, true)
        } else if can_use_64_tile {
            (64u32, 32u32, 64u32, true)
        } else {
            (32u32, 32u32, 32u32, configuration.output_dim % 32 == 0)
        };

        let big = <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel::new(
            context,
            configuration.data_type,
            group_size,
            bits,
            bm,
            bk,
            bn,
            2u32,
            2u32,
            use_zero_points,
            use_mlx_quant,
            configuration.use_hadamard,
            aligned_n,
        )
        .map_err(QuantizedMatmulError::BackendError)?;

        let matrix_matrix = if aligned_n_32 && !configuration.use_hadamard {
            let small = <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel::new(
                context,
                configuration.data_type,
                group_size,
                bits,
                8u32,
                32u32,
                32u32,
                1u32,
                1u32,
                use_zero_points,
                use_mlx_quant,
                false,
                true,
            )
            .map_err(QuantizedMatmulError::BackendError)?;
            MatrixMatrixKernel::BigAndSmall {
                big,
                small,
            }
        } else {
            MatrixMatrixKernel::Big(big)
        };

        Ok(Self {
            matrix_vector,
            matrix_matrix,
            input_dim: configuration.input_dim,
            output_dim: configuration.output_dim,
            quantization_type: configuration.quantization_type,
        })
    }

    pub fn encode(
        &self,
        encoder: &mut Encoder<B>,
        arguments: QuantizedMatmulArguments<B>,
    ) {
        let QuantizedMatmulArguments {
            a,
            b,
            scales,
            zero_points_or_biases,
            output,
            hadamard_factors,
            batch_dim,
        } = arguments;

        let (zero_points, biases) = match self.quantization_type {
            QuantizedMatmulType::ZeroPoint => (Some(zero_points_or_biases), None),
            QuantizedMatmulType::Mlx => (None, Some(zero_points_or_biases)),
        };

        macro_rules! encode_kernel {
            ($kernel:expr $(, $hadamard:expr)?) => {
                $kernel.encode(
                    b,
                    scales,
                    zero_points,
                    biases,
                    a,
                    output,
                    $($hadamard,)?
                    self.input_dim as u32,
                    self.output_dim as u32,
                    batch_dim as u32,
                    encoder,
                )
            };
        }

        if batch_dim >= 8 && self.output_dim > 1 {
            if let Some(kernel) = self.matrix_matrix.pick(batch_dim) {
                encode_kernel!(kernel, hadamard_factors);
                return;
            }
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
