use std::collections::HashMap;

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder, Kernels,
        gpu_types::QuantizationMode,
        kernel::{
            QuantizedMatmulQmmKernel, QuantizedMatmulQmmTransposed64x64Kernel, QuantizedMatmulQmmTransposedKernel,
            QuantizedMatmulQmmTransposedSmallKernel, QuantizedMatmulQmmTransposedSmallSplitKPartialKernel,
            QuantizedMatmulQmmTransposedSmallSplitKReduceKernel, QuantizedMatmulQmmTransposedWideKernel,
            QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel, QuantizedMatmulQvmKernel,
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
    #[error("Unsupported bits: {0}")]
    UnsupportedBits(usize),
    #[error("Value `{name}` does not fit i32: {value}")]
    ValueOutOfRange {
        name: &'static str,
        value: usize,
    },
    #[error("Quantization type mismatch: kernel={kernel:?}, args={args:?}")]
    QuantizationTypeMismatch {
        kernel: QuantizedMatmulType,
        args: QuantizedMatmulType,
    },
    #[error("Missing kernel for key: {0}")]
    MissingKernel(&'static str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizedMatmulType {
    ZeroPoint,
    Mlx,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ForceKernel {
    #[default]
    Auto,
    QmvFast,
    QmmTransposedSmall,
    QmmTransposedSmallSplitK,
    QmmTransposed64x64,
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
    pub force_kernel: ForceKernel,
}

pub struct QuantizedMatmulArguments<'a, B: Backend> {
    pub a_buffer: &'a B::Buffer,
    pub a_offset: usize,
    pub b_buffer: &'a B::Buffer,
    pub scales_buffer: &'a B::Buffer,
    pub zero_points_or_biases_buffer: &'a B::Buffer,
    pub output_buffer: &'a mut B::Buffer,
    pub batch: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub quantization_type: QuantizedMatmulType,
}

pub struct QuantizedMatmulKernelEncodable<B: Backend> {
    kernels: HashMap<KernelKey, EncodableVariant<B>>,
    matrix_vector_key: KernelKey,
    matrix_matrix_key: KernelKey,
    output_dim: usize,
    group_size: usize,
    quantization_type: QuantizedMatmulType,
    force_kernel: ForceKernel,
}

enum EncodableVariant<B: Backend> {
    Qmv(<B::Kernels as Kernels>::QuantizedMatmulQmvKernel),
    QmvFast(<B::Kernels as Kernels>::QuantizedMatmulQmvFastKernel),
    Qvm(<B::Kernels as Kernels>::QuantizedMatmulQvmKernel),
    Qmm(<B::Kernels as Kernels>::QuantizedMatmulQmmKernel),
    QmmTransposed(<B::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel),
    QmmTransposed64x64(<B::Kernels as Kernels>::QuantizedMatmulQmmTransposed64x64Kernel),
    QmmTransposedSmall(<B::Kernels as Kernels>::QuantizedMatmulQmmTransposedSmallKernel),
    QmmTransposedSmallSplitK {
        partial: <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedSmallSplitKPartialKernel,
        reduce: <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedSmallSplitKReduceKernel,
    },
    QmmTransposedWide(<B::Kernels as Kernels>::QuantizedMatmulQmmTransposedWideKernel),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum KernelKey {
    MatrixVector(MatrixVectorFamily),
    MatrixMatrix(MatrixMatrixFamily),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MatrixVectorFamily {
    Qmv,
    QmvFast,
    Qvm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MatrixMatrixFamily {
    QmmAlignedK,
    QmmUnalignedK,
    QmmTransposedAlignedN,
    QmmTransposedUnalignedN,
    QmmTransposed64x64,
    QmmTransposedSmall,
    QmmTransposedSmallSplitK,
    QmmTransposedWide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeVariant {
    MatrixVector,
    MatrixMatrix,
}

impl<B: Backend> QuantizedMatmulKernelEncodable<B> {
    pub fn new(
        context: &B::Context,
        configuration: QuantizedMatmulConfiguration,
    ) -> Result<Self, QuantizedMatmulError<B>> {
        validate_configuration(&configuration)?;

        let bits = quant_bits(configuration.mode)?;
        let use_mlx_quant = matches!(configuration.quantization_type, QuantizedMatmulType::Mlx);

        let force_kernel = configuration.force_kernel;

        let matrix_vector_family = match force_kernel {
            ForceKernel::QmvFast => MatrixVectorFamily::QmvFast,
            _ => select_matrix_vector_family(&configuration),
        };
        let matrix_matrix_family = match force_kernel {
            ForceKernel::QmmTransposedSmall => MatrixMatrixFamily::QmmTransposedSmall,
            ForceKernel::QmmTransposedSmallSplitK => MatrixMatrixFamily::QmmTransposedSmallSplitK,
            ForceKernel::QmmTransposed64x64 => MatrixMatrixFamily::QmmTransposed64x64,
            _ => select_matrix_matrix_family(&configuration, bits),
        };
        let matrix_vector_key = KernelKey::MatrixVector(matrix_vector_family);
        let matrix_matrix_key = KernelKey::MatrixMatrix(matrix_matrix_family);

        let mut kernels = HashMap::new();
        match force_kernel {
            ForceKernel::QmvFast => {
                kernels.insert(
                    matrix_vector_key,
                    create_matrix_vector_kernel(
                        context,
                        configuration.data_type,
                        configuration.group_size,
                        bits,
                        use_mlx_quant,
                        matrix_vector_family,
                    )?,
                );
            },
            ForceKernel::QmmTransposedSmall
            | ForceKernel::QmmTransposedSmallSplitK
            | ForceKernel::QmmTransposed64x64 => {
                kernels.insert(
                    matrix_matrix_key,
                    create_matrix_matrix_kernel(
                        context,
                        configuration.data_type,
                        configuration.group_size,
                        bits,
                        use_mlx_quant,
                        matrix_matrix_family,
                    )?,
                );
            },
            ForceKernel::Auto => {
                kernels.insert(
                    matrix_vector_key,
                    create_matrix_vector_kernel(
                        context,
                        configuration.data_type,
                        configuration.group_size,
                        bits,
                        use_mlx_quant,
                        matrix_vector_family,
                    )?,
                );
                kernels.insert(
                    matrix_matrix_key,
                    create_matrix_matrix_kernel(
                        context,
                        configuration.data_type,
                        configuration.group_size,
                        bits,
                        use_mlx_quant,
                        matrix_matrix_family,
                    )?,
                );
            },
        }

        Ok(Self {
            kernels,
            matrix_vector_key,
            matrix_matrix_key,
            output_dim: configuration.output_dim,
            group_size: configuration.group_size,
            quantization_type: configuration.quantization_type,
            force_kernel,
        })
    }

    pub fn encode(
        &self,
        encoder: &mut Encoder<B>,
        arguments: QuantizedMatmulArguments<B>,
    ) -> Result<(), QuantizedMatmulError<B>> {
        if arguments.quantization_type != self.quantization_type {
            return Err(QuantizedMatmulError::QuantizationTypeMismatch {
                kernel: self.quantization_type,
                args: arguments.quantization_type,
            });
        }

        let key = match self.select_runtime_variant(arguments.batch) {
            RuntimeVariant::MatrixVector => self.matrix_vector_key,
            RuntimeVariant::MatrixMatrix => self.matrix_matrix_key,
        };

        let kernel = self.kernels.get(&key).ok_or(QuantizedMatmulError::MissingKernel(kernel_key_name(key)))?;
        let k = to_i32("input_dim", arguments.input_dim)?;
        let n = to_i32("output_dim", arguments.output_dim)?;
        let m = to_i32("batch", arguments.batch)?;
        let group_size = self.group_size;
        let (zero_points, biases) = quant_buffers::<B>(arguments.zero_points_or_biases_buffer, self.quantization_type);
        let a_with_offset = (arguments.a_buffer, arguments.a_offset);

        match kernel {
            EncodableVariant::Qmv(kernel) => {
                kernel.encode(
                    arguments.b_buffer,
                    arguments.scales_buffer,
                    zero_points,
                    biases,
                    a_with_offset,
                    arguments.output_buffer,
                    k,
                    n,
                    m,
                    encoder,
                );
            },
            EncodableVariant::QmvFast(kernel) => {
                kernel.encode(
                    arguments.b_buffer,
                    arguments.scales_buffer,
                    zero_points,
                    biases,
                    a_with_offset,
                    arguments.output_buffer,
                    k,
                    n,
                    m,
                    encoder,
                );
            },
            EncodableVariant::Qvm(kernel) => {
                kernel.encode(
                    arguments.b_buffer,
                    arguments.scales_buffer,
                    zero_points,
                    biases,
                    a_with_offset,
                    arguments.output_buffer,
                    k,
                    n,
                    m,
                    encoder,
                );
            },
            EncodableVariant::Qmm(kernel) => {
                kernel.encode(
                    arguments.b_buffer,
                    arguments.scales_buffer,
                    zero_points,
                    biases,
                    a_with_offset,
                    arguments.output_buffer,
                    k,
                    n,
                    m,
                    encoder,
                );
            },
            EncodableVariant::QmmTransposed(kernel) => {
                kernel.encode(
                    arguments.b_buffer,
                    arguments.scales_buffer,
                    zero_points,
                    biases,
                    a_with_offset,
                    arguments.output_buffer,
                    k,
                    n,
                    m,
                    encoder,
                );
            },
            EncodableVariant::QmmTransposed64x64(kernel) => {
                kernel.encode(
                    arguments.b_buffer,
                    arguments.scales_buffer,
                    zero_points,
                    biases,
                    a_with_offset,
                    arguments.output_buffer,
                    k,
                    n,
                    m,
                    encoder,
                );
            },
            EncodableVariant::QmmTransposedSmall(kernel) => {
                kernel.encode(
                    arguments.b_buffer,
                    arguments.scales_buffer,
                    zero_points,
                    biases,
                    a_with_offset,
                    arguments.output_buffer,
                    k,
                    n,
                    m,
                    encoder,
                );
            },
            EncodableVariant::QmmTransposedSmallSplitK {
                partial: partial_kernel,
                reduce: reduce_kernel,
            } => {
                // MLX heuristic for split_k
                let current_tgs = ((n + 31) / 32) * ((m + 7) / 8);
                let split_k_raw = std::cmp::max(1i32, 512 / std::cmp::max(current_tgs, 1));
                let k_per_group = to_i32("group_size", group_size)?;
                // Can't split finer than group_size
                let max_split_k = k / k_per_group;
                let mut split_k = std::cmp::min(split_k_raw, max_split_k);
                // Adjust until k % (split_k * group_size) == 0
                while split_k > 1 && k % (split_k * k_per_group) != 0 {
                    split_k -= 1;
                }
                let split_k = std::cmp::max(1, split_k);

                // Allocate temporary buffer: split_k * m * n elements of T (bfloat = 2 bytes)
                let elem_size = 2usize; // bfloat16
                let partial_size = (split_k as usize) * (m as usize) * (n as usize) * elem_size;
                let mut partial_buf =
                    encoder.allocate_scratch(partial_size).map_err(QuantizedMatmulError::BackendError)?;

                partial_kernel.encode(
                    arguments.b_buffer,
                    arguments.scales_buffer,
                    zero_points,
                    biases,
                    a_with_offset,
                    &mut partial_buf,
                    k,
                    n,
                    m,
                    split_k,
                    encoder,
                );

                reduce_kernel.encode(&partial_buf, arguments.output_buffer, n, m, split_k, encoder);
            },
            EncodableVariant::QmmTransposedWide(kernel) => {
                kernel.encode(
                    arguments.b_buffer,
                    arguments.scales_buffer,
                    zero_points,
                    biases,
                    a_with_offset,
                    arguments.output_buffer,
                    k,
                    n,
                    m,
                    encoder,
                );
            },
        }

        Ok(())
    }

    fn select_runtime_variant(
        &self,
        batch: usize,
    ) -> RuntimeVariant {
        match self.force_kernel {
            ForceKernel::QmvFast => RuntimeVariant::MatrixVector,
            ForceKernel::QmmTransposedSmall
            | ForceKernel::QmmTransposedSmallSplitK
            | ForceKernel::QmmTransposed64x64 => RuntimeVariant::MatrixMatrix,
            ForceKernel::Auto => {
                if batch < 4 || self.output_dim == 1 {
                    RuntimeVariant::MatrixVector
                } else {
                    RuntimeVariant::MatrixMatrix
                }
            },
        }
    }
}

fn create_matrix_vector_kernel<B: Backend>(
    context: &B::Context,
    data_type: DataType,
    group_size: usize,
    bits: usize,
    use_mlx_quant: bool,
    family: MatrixVectorFamily,
) -> Result<EncodableVariant<B>, QuantizedMatmulError<B>> {
    let group_size = to_i32("group_size", group_size)?;
    let bits = to_i32("bits", bits)?;
    let use_zero_points = !use_mlx_quant;

    let kernel = match family {
        MatrixVectorFamily::Qmv => EncodableVariant::Qmv(
            <B::Kernels as Kernels>::QuantizedMatmulQmvKernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
            )
            .map_err(QuantizedMatmulError::BackendError)?,
        ),
        MatrixVectorFamily::QmvFast => EncodableVariant::QmvFast(
            <B::Kernels as Kernels>::QuantizedMatmulQmvFastKernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
            )
            .map_err(QuantizedMatmulError::BackendError)?,
        ),
        MatrixVectorFamily::Qvm => EncodableVariant::Qvm(
            <B::Kernels as Kernels>::QuantizedMatmulQvmKernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
            )
            .map_err(QuantizedMatmulError::BackendError)?,
        ),
    };

    Ok(kernel)
}

fn create_matrix_matrix_kernel<B: Backend>(
    context: &B::Context,
    data_type: DataType,
    group_size: usize,
    bits: usize,
    use_mlx_quant: bool,
    family: MatrixMatrixFamily,
) -> Result<EncodableVariant<B>, QuantizedMatmulError<B>> {
    let group_size = to_i32("group_size", group_size)?;
    let bits = to_i32("bits", bits)?;
    let use_zero_points = !use_mlx_quant;

    let kernel = match family {
        MatrixMatrixFamily::QmmAlignedK => EncodableVariant::Qmm(
            <B::Kernels as Kernels>::QuantizedMatmulQmmKernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
                true,
            )
            .map_err(QuantizedMatmulError::BackendError)?,
        ),
        MatrixMatrixFamily::QmmUnalignedK => EncodableVariant::Qmm(
            <B::Kernels as Kernels>::QuantizedMatmulQmmKernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
                false,
            )
            .map_err(QuantizedMatmulError::BackendError)?,
        ),
        MatrixMatrixFamily::QmmTransposedAlignedN => EncodableVariant::QmmTransposed(
            <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
                true,
            )
            .map_err(QuantizedMatmulError::BackendError)?,
        ),
        MatrixMatrixFamily::QmmTransposedUnalignedN => EncodableVariant::QmmTransposed(
            <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
                false,
            )
            .map_err(QuantizedMatmulError::BackendError)?,
        ),
        MatrixMatrixFamily::QmmTransposedSmall => EncodableVariant::QmmTransposedSmall(
            <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedSmallKernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
            )
            .map_err(QuantizedMatmulError::BackendError)?,
        ),
        MatrixMatrixFamily::QmmTransposedSmallSplitK => {
            let partial = <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedSmallSplitKPartialKernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
            )
            .map_err(QuantizedMatmulError::BackendError)?;
            let reduce =
                <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedSmallSplitKReduceKernel::new(context, data_type)
                    .map_err(QuantizedMatmulError::BackendError)?;
            EncodableVariant::QmmTransposedSmallSplitK {
                partial,
                reduce,
            }
        },
        MatrixMatrixFamily::QmmTransposed64x64 => EncodableVariant::QmmTransposed64x64(
            <B::Kernels as Kernels>::QuantizedMatmulQmmTransposed64x64Kernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
            )
            .map_err(QuantizedMatmulError::BackendError)?,
        ),
        MatrixMatrixFamily::QmmTransposedWide => EncodableVariant::QmmTransposedWide(
            <B::Kernels as Kernels>::QuantizedMatmulQmmTransposedWideKernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
            )
            .map_err(QuantizedMatmulError::BackendError)?,
        ),
    };

    Ok(kernel)
}

fn validate_configuration<B: Backend>(
    configuration: &QuantizedMatmulConfiguration
) -> Result<(), QuantizedMatmulError<B>> {
    if !matches!(configuration.data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(QuantizedMatmulError::UnsupportedDataType(configuration.data_type));
    }

    if !matches!(configuration.group_size, 32 | 64 | 128) {
        return Err(QuantizedMatmulError::UnsupportedGroupSize(configuration.group_size));
    }

    let _ = quant_bits(configuration.mode)?;
    Ok(())
}

fn select_matrix_vector_family(configuration: &QuantizedMatmulConfiguration) -> MatrixVectorFamily {
    if configuration.weights_transposed {
        if configuration.output_dim % 8 == 0 && configuration.input_dim % 512 == 0 {
            MatrixVectorFamily::QmvFast
        } else {
            MatrixVectorFamily::Qmv
        }
    } else {
        MatrixVectorFamily::Qvm
    }
}

fn select_matrix_matrix_family(
    configuration: &QuantizedMatmulConfiguration,
    bits: usize,
) -> MatrixMatrixFamily {
    if configuration.weights_transposed {
        let aligned_n = configuration.output_dim % 32 == 0;
        let aligned_n_64 = configuration.output_dim % 64 == 0;
        let use_small = aligned_n && configuration.data_type == DataType::BF16 && matches!(bits, 4 | 8);
        let use_64x64 = aligned_n_64
            && configuration.data_type == DataType::BF16
            && matches!(configuration.group_size, 64 | 128)
            && matches!(bits, 4 | 8);
        let use_wide = aligned_n_64 && configuration.data_type == DataType::BF16 && matches!(bits, 4 | 8) && !use_64x64;
        if use_small {
            MatrixMatrixFamily::QmmTransposedSmall
        } else if use_64x64 {
            MatrixMatrixFamily::QmmTransposed64x64
        } else if use_wide {
            MatrixMatrixFamily::QmmTransposedWide
        } else if aligned_n {
            MatrixMatrixFamily::QmmTransposedAlignedN
        } else {
            MatrixMatrixFamily::QmmTransposedUnalignedN
        }
    } else if configuration.input_dim % 32 == 0 {
        MatrixMatrixFamily::QmmAlignedK
    } else {
        MatrixMatrixFamily::QmmUnalignedK
    }
}

fn quant_bits<B: Backend>(mode: QuantizationMode) -> Result<usize, QuantizedMatmulError<B>> {
    let bits = match mode {
        QuantizationMode::UINT4 => 4,
        QuantizationMode::INT8 | QuantizationMode::UINT8 => 8,
    };
    if matches!(bits, 4 | 8) {
        Ok(bits)
    } else {
        Err(QuantizedMatmulError::UnsupportedBits(bits))
    }
}

fn to_i32<B: Backend>(
    name: &'static str,
    value: usize,
) -> Result<i32, QuantizedMatmulError<B>> {
    i32::try_from(value).map_err(|_| QuantizedMatmulError::ValueOutOfRange {
        name,
        value,
    })
}

fn quant_buffers<'a, B: Backend>(
    buffer: &'a B::Buffer,
    quantization_type: QuantizedMatmulType,
) -> (Option<&'a B::Buffer>, Option<&'a B::Buffer>) {
    match quantization_type {
        QuantizedMatmulType::ZeroPoint => (Some(buffer), None),
        QuantizedMatmulType::Mlx => (None, Some(buffer)),
    }
}

fn kernel_key_name(key: KernelKey) -> &'static str {
    match key {
        KernelKey::MatrixVector(MatrixVectorFamily::Qmv) => "matrix_vector_qmv",
        KernelKey::MatrixVector(MatrixVectorFamily::QmvFast) => "matrix_vector_qmv_fast",
        KernelKey::MatrixVector(MatrixVectorFamily::Qvm) => "matrix_vector_qvm",
        KernelKey::MatrixMatrix(MatrixMatrixFamily::QmmAlignedK) => "matrix_matrix_qmm_aligned_k",
        KernelKey::MatrixMatrix(MatrixMatrixFamily::QmmUnalignedK) => "matrix_matrix_qmm_unaligned_k",
        KernelKey::MatrixMatrix(MatrixMatrixFamily::QmmTransposedAlignedN) => "matrix_matrix_qmm_transposed_aligned_n",
        KernelKey::MatrixMatrix(MatrixMatrixFamily::QmmTransposedUnalignedN) => {
            "matrix_matrix_qmm_transposed_unaligned_n"
        },
        KernelKey::MatrixMatrix(MatrixMatrixFamily::QmmTransposedSmall) => "matrix_matrix_qmm_transposed_small",
        KernelKey::MatrixMatrix(MatrixMatrixFamily::QmmTransposedSmallSplitK) => {
            "matrix_matrix_qmm_transposed_small_splitk"
        },
        KernelKey::MatrixMatrix(MatrixMatrixFamily::QmmTransposed64x64) => "matrix_matrix_qmm_transposed_64x64",
        KernelKey::MatrixMatrix(MatrixMatrixFamily::QmmTransposedWide) => "matrix_matrix_qmm_transposed_wide",
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/backends/common/kernel/quant_matmul_test.rs"]
mod tests;
