use crate::{
    DataType,
    backends::common::{
        Backend, Encoder, Kernels,
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
    pub a_buffer: &'a B::Buffer,
    pub a_offset: usize,
    pub b_buffer: &'a B::Buffer,
    pub scales_buffer: &'a B::Buffer,
    pub zero_points_or_biases_buffer: &'a B::Buffer,
    pub output_buffer: &'a mut B::Buffer,
    pub hadamard_factors: Option<&'a B::Buffer>,
    pub batch_dim: usize,
}

pub struct QuantizedMatmulKernelEncodable<B: Backend> {
    matrix_vector: MatrixVectorKernel<B>,
    matrix_matrix: <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel,
    input_dim: usize,
    output_dim: usize,
    quantization_type: QuantizedMatmulType,
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

        // 64-tile (BM=BN=64): bf16 + output_dim%64==0; BK=64 if group_size≥64, BK=32 otherwise.
        // 32-tile fallback otherwise.
        let aligned_n_64 = configuration.output_dim % 64 == 0;
        let is_bf16 = configuration.data_type == DataType::BF16;
        let can_use_64_tile = aligned_n_64 && is_bf16;

        let (bm, bk, bn, aligned_n) = if can_use_64_tile && matches!(configuration.group_size, 64 | 128) {
            (64u32, 64u32, 64u32, true)
        } else if can_use_64_tile {
            (64u32, 32u32, 64u32, true)
        } else {
            (32u32, 32u32, 32u32, configuration.output_dim % 32 == 0)
        };

        let matrix_matrix = <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel::new(
            context,
            configuration.data_type,
            group_size,
            bits,
            bm,
            bk,
            bn,
            use_zero_points,
            use_mlx_quant,
            configuration.use_hadamard,
            aligned_n,
        )
        .map_err(QuantizedMatmulError::BackendError)?;

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
    ) -> Result<(), QuantizedMatmulError<B>> {
        let (zero_points, biases) = match self.quantization_type {
            QuantizedMatmulType::ZeroPoint => (Some(arguments.zero_points_or_biases_buffer), None),
            QuantizedMatmulType::Mlx => (None, Some(arguments.zero_points_or_biases_buffer)),
        };

        macro_rules! encode_kernel {
            ($kernel:expr, $($hadamard:expr)?) => {
                $kernel.encode(
                    arguments.b_buffer,
                    arguments.scales_buffer,
                    zero_points,
                    biases,
                    (arguments.a_buffer, arguments.a_offset),
                    arguments.output_buffer,
                    $($hadamard,)?
                    self.input_dim as u32,
                    self.output_dim as u32,
                    arguments.batch_dim as u32,
                    encoder,
                )
            };
        }

        if arguments.batch_dim < 32 || self.output_dim == 1 {
            match &self.matrix_vector {
                MatrixVectorKernel::Qmv(k) => encode_kernel!(k,),
                MatrixVectorKernel::QmvFast(k) => encode_kernel!(k, arguments.hadamard_factors),
            }
        } else {
            encode_kernel!(&self.matrix_matrix, arguments.hadamard_factors);
        }

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/backends/common/kernel/quant_matmul_test.rs"]
mod tests;
