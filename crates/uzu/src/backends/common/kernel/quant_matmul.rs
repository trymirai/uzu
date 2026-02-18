use std::collections::HashMap;

use crate::{
    DataType,
    backends::common::{
        Backend, Kernels,
        kernel::{
            QuantizedMatmulQmmKernel, QuantizedMatmulQmmTransposed64x64Kernel, QuantizedMatmulQmmTransposedKernel,
            QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel, QuantizedMatmulQvmKernel,
        },
    },
    config::QuantizationMode,
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

pub struct QuantizedMatmulKernelEncodable<B: Backend> {
    kernels: HashMap<KernelKey, EncodableVariant<B>>,
    matrix_vector_key: KernelKey,
    matrix_matrix_key: KernelKey,
    output_dim: usize,
    quantization_type: QuantizedMatmulType,
}

enum EncodableVariant<B: Backend> {
    Qmv(<B::Kernels as Kernels>::QuantizedMatmulQmvKernel),
    QmvFast(<B::Kernels as Kernels>::QuantizedMatmulQmvFastKernel),
    Qvm(<B::Kernels as Kernels>::QuantizedMatmulQvmKernel),
    Qmm(<B::Kernels as Kernels>::QuantizedMatmulQmmKernel),
    QmmTransposed(<B::Kernels as Kernels>::QuantizedMatmulQmmTransposedKernel),
    QmmTransposed64x64(<B::Kernels as Kernels>::QuantizedMatmulQmmTransposed64x64Kernel),
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

        let matrix_vector_family = select_matrix_vector_family(&configuration);
        let matrix_matrix_family = select_matrix_matrix_family(&configuration, bits);
        let matrix_vector_key = KernelKey::MatrixVector(matrix_vector_family);
        let matrix_matrix_key = KernelKey::MatrixMatrix(matrix_matrix_family);

        let mut kernels = HashMap::new();
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

        Ok(Self {
            kernels,
            matrix_vector_key,
            matrix_matrix_key,
            output_dim: configuration.output_dim,
            quantization_type: configuration.quantization_type,
        })
    }

    pub fn encode(
        &self,
        encoder: &B::ComputeEncoder,
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
        }

        Ok(())
    }

    fn select_runtime_variant(
        &self,
        batch: usize,
    ) -> RuntimeVariant {
        if batch < 32 || self.output_dim == 1 {
            RuntimeVariant::MatrixVector
        } else {
            RuntimeVariant::MatrixMatrix
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
        let use_64x64 = aligned_n
            && configuration.data_type == DataType::BF16
            && matches!(configuration.group_size, 64 | 128)
            && matches!(bits, 4 | 8);
        if use_64x64 {
            MatrixMatrixFamily::QmmTransposed64x64
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
        QuantizationMode::UInt4 => 4,
        QuantizationMode::UInt8 | QuantizationMode::Int8 => 8,
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
    buffer: &'a B::NativeBuffer,
    quantization_type: QuantizedMatmulType,
) -> (Option<&'a B::NativeBuffer>, Option<&'a B::NativeBuffer>) {
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
        KernelKey::MatrixMatrix(MatrixMatrixFamily::QmmTransposed64x64) => "matrix_matrix_qmm_transposed_64x64",
    }
}
