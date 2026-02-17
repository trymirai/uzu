use std::collections::HashMap;

use metal::{MTLBuffer, MTLComputeCommandEncoder};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::{
    DataType,
    backends::{
        common::kernel::{
            QuantizedMatmulQmmKernel, QuantizedMatmulQmmTransposed64x64Kernel, QuantizedMatmulQmmTransposedKernel,
            QuantizedMatmulQmvFastKernel, QuantizedMatmulQmvKernel, QuantizedMatmulQvmKernel,
            matmul::{
                QuantizedMatmulArguments as GenericQuantizedMatmulArguments, QuantizedMatmulConfiguration,
                QuantizedMatmulKernel as QuantizedMatmulKernelTrait, QuantizedMatmulType,
            },
        },
        metal::{
            Metal, MetalContext, MetalError,
            kernel::dsl::{
                QuantizedMatmulQmmMetalKernel, QuantizedMatmulQmmTransposed64x64MetalKernel,
                QuantizedMatmulQmmTransposedMetalKernel, QuantizedMatmulQmvFastMetalKernel,
                QuantizedMatmulQmvMetalKernel, QuantizedMatmulQvmMetalKernel,
            },
        },
    },
    config::QuantizationMode,
};

#[derive(Debug, thiserror::Error)]
pub enum QuantizedMatmulError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MetalError),
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

pub struct QuantizedMatmulKernel {
    kernels: HashMap<KernelKey, EncodableVariant>,
    matrix_vector_key: KernelKey,
    matrix_matrix_key: KernelKey,
    output_dim: usize,
    quantization_type: QuantizedMatmulType,
}

enum EncodableVariant {
    Qmv(QuantizedMatmulQmvMetalKernel),
    QmvFast(QuantizedMatmulQmvFastMetalKernel),
    Qvm(QuantizedMatmulQvmMetalKernel),
    Qmm(QuantizedMatmulQmmMetalKernel),
    QmmTransposed(QuantizedMatmulQmmTransposedMetalKernel),
    QmmTransposed64x64(QuantizedMatmulQmmTransposed64x64MetalKernel),
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

impl QuantizedMatmulKernel {
    pub fn new(
        context: &MetalContext,
        configuration: QuantizedMatmulConfiguration,
    ) -> Result<Self, QuantizedMatmulError> {
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
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: GenericQuantizedMatmulArguments<Metal>,
    ) -> Result<(), QuantizedMatmulError> {
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

        let kernel = self
            .kernels
            .get(&key)
            .ok_or(QuantizedMatmulError::MissingKernel(kernel_key_name(key)))?;
        let k = to_i32("input_dim", arguments.input_dim)?;
        let n = to_i32("output_dim", arguments.output_dim)?;
        let m = to_i32("batch", arguments.batch)?;
        let (zero_points, biases) = quant_buffers(arguments.zero_points_or_biases_buffer, self.quantization_type);
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

impl QuantizedMatmulKernelTrait for QuantizedMatmulKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        configuration: QuantizedMatmulConfiguration,
    ) -> Result<Self, MetalError> {
        QuantizedMatmulKernel::new(context, configuration).map_err(|error| MetalError::Generic(format!("{error:?}")))
    }

    fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        arguments: GenericQuantizedMatmulArguments<Metal>,
    ) {
        QuantizedMatmulKernel::encode(self, encoder, arguments).expect("Failed to encode quantized matmul");
    }
}

fn create_matrix_vector_kernel(
    context: &MetalContext,
    data_type: DataType,
    group_size: usize,
    bits: usize,
    use_mlx_quant: bool,
    family: MatrixVectorFamily,
) -> Result<EncodableVariant, QuantizedMatmulError> {
    let group_size = to_i32("group_size", group_size)?;
    let bits = to_i32("bits", bits)?;
    let use_zero_points = !use_mlx_quant;

    let kernel = match family {
        MatrixVectorFamily::Qmv => EncodableVariant::Qmv(QuantizedMatmulQmvMetalKernel::new(
            context,
            data_type,
            group_size,
            bits,
            use_zero_points,
            use_mlx_quant,
        )?),
        MatrixVectorFamily::QmvFast => EncodableVariant::QmvFast(QuantizedMatmulQmvFastMetalKernel::new(
            context,
            data_type,
            group_size,
            bits,
            use_zero_points,
            use_mlx_quant,
        )?),
        MatrixVectorFamily::Qvm => EncodableVariant::Qvm(QuantizedMatmulQvmMetalKernel::new(
            context,
            data_type,
            group_size,
            bits,
            use_zero_points,
            use_mlx_quant,
        )?),
    };

    Ok(kernel)
}

fn create_matrix_matrix_kernel(
    context: &MetalContext,
    data_type: DataType,
    group_size: usize,
    bits: usize,
    use_mlx_quant: bool,
    family: MatrixMatrixFamily,
) -> Result<EncodableVariant, QuantizedMatmulError> {
    let group_size = to_i32("group_size", group_size)?;
    let bits = to_i32("bits", bits)?;
    let use_zero_points = !use_mlx_quant;

    let kernel = match family {
        MatrixMatrixFamily::QmmAlignedK => EncodableVariant::Qmm(QuantizedMatmulQmmMetalKernel::new(
            context,
            data_type,
            group_size,
            bits,
            use_zero_points,
            use_mlx_quant,
            true,
        )?),
        MatrixMatrixFamily::QmmUnalignedK => EncodableVariant::Qmm(QuantizedMatmulQmmMetalKernel::new(
            context,
            data_type,
            group_size,
            bits,
            use_zero_points,
            use_mlx_quant,
            false,
        )?),
        MatrixMatrixFamily::QmmTransposedAlignedN => EncodableVariant::QmmTransposed(
            QuantizedMatmulQmmTransposedMetalKernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
                true,
            )?,
        ),
        MatrixMatrixFamily::QmmTransposedUnalignedN => EncodableVariant::QmmTransposed(
            QuantizedMatmulQmmTransposedMetalKernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
                false,
            )?,
        ),
        MatrixMatrixFamily::QmmTransposed64x64 => EncodableVariant::QmmTransposed64x64(
            QuantizedMatmulQmmTransposed64x64MetalKernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
            )?,
        ),
    };

    Ok(kernel)
}

fn validate_configuration(configuration: &QuantizedMatmulConfiguration) -> Result<(), QuantizedMatmulError> {
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

fn quant_bits(mode: QuantizationMode) -> Result<usize, QuantizedMatmulError> {
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

fn to_i32(
    name: &'static str,
    value: usize,
) -> Result<i32, QuantizedMatmulError> {
    i32::try_from(value).map_err(|_| QuantizedMatmulError::ValueOutOfRange {
        name,
        value,
    })
}

fn quant_buffers<'a>(
    buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    quantization_type: QuantizedMatmulType,
) -> (
    Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
) {
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

