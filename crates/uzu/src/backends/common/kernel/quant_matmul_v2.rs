use std::collections::HashMap;

use crate::{
    DataType,
    backends::common::{
        Backend, CommandBuffer, Kernels,
        kernel::{
            QuantizedMatmulGemmTransposed64x64V2Kernel, QuantizedMatmulGemmTransposedV2Kernel,
            QuantizedMatmulGemmV2Kernel, QuantizedMatmulGemvFastV2Kernel, QuantizedMatmulGemvV2Kernel,
            QuantizedMatmulVectorMatrixV2Kernel,
        },
    },
    config::QuantizationMode,
};

use super::quant_matmul::{
    QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError,
    QuantizedMatmulKernelEncodable, QuantizedMatmulType,
};

pub struct QuantizedMatmulKernelEncodableV2<B: Backend> {
    kernels: HashMap<KernelKey, EncodableVariant<B>>,
    matrix_vector_key: KernelKey,
    matrix_matrix_key: KernelKey,
    output_dim: usize,
    quantization_type: QuantizedMatmulType,
}

enum EncodableVariant<B: Backend> {
    Gemv(<B::Kernels as Kernels>::QuantizedMatmulGemvV2Kernel),
    GemvFast(<B::Kernels as Kernels>::QuantizedMatmulGemvFastV2Kernel),
    VectorMatrix(<B::Kernels as Kernels>::QuantizedMatmulVectorMatrixV2Kernel),
    Gemm(<B::Kernels as Kernels>::QuantizedMatmulGemmV2Kernel),
    GemmTransposed(<B::Kernels as Kernels>::QuantizedMatmulGemmTransposedV2Kernel),
    GemmTransposed64x64(<B::Kernels as Kernels>::QuantizedMatmulGemmTransposed64x64V2Kernel),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum KernelKey {
    MatrixVector(MatrixVectorFamily),
    MatrixMatrix(MatrixMatrixFamily),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MatrixVectorFamily {
    Gemv,
    GemvFast,
    VectorMatrix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MatrixMatrixFamily {
    GemmAlignedK,
    GemmUnalignedK,
    GemmTransposedAlignedN,
    GemmTransposedUnalignedN,
    GemmTransposed64x64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeVariant {
    MatrixVector,
    MatrixMatrix,
}

impl<B: Backend> QuantizedMatmulKernelEncodableV2<B> {
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
        encoder: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
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

        let kernel = self
            .kernels
            .get(&key)
            .ok_or(QuantizedMatmulError::MissingKernel(kernel_key_name(key)))?;
        let k = to_i32("input_dim", arguments.input_dim)?;
        let n = to_i32("output_dim", arguments.output_dim)?;
        let m = to_i32("batch", arguments.batch)?;
        let (zero_points, biases) =
            quant_buffers::<B>(arguments.zero_points_or_biases_buffer, self.quantization_type);
        let a_with_offset = (arguments.a_buffer, arguments.a_offset);

        match kernel {
            EncodableVariant::Gemv(kernel) => {
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
            }
            EncodableVariant::GemvFast(kernel) => {
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
            }
            EncodableVariant::VectorMatrix(kernel) => {
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
            }
            EncodableVariant::Gemm(kernel) => {
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
            }
            EncodableVariant::GemmTransposed(kernel) => {
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
            }
            EncodableVariant::GemmTransposed64x64(kernel) => {
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
            }
        }

        Ok(())
    }

    fn select_runtime_variant(&self, batch: usize) -> RuntimeVariant {
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
        MatrixVectorFamily::Gemv => EncodableVariant::Gemv(
            <B::Kernels as Kernels>::QuantizedMatmulGemvV2Kernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
            )
            .map_err(QuantizedMatmulError::BackendError)?,
        ),
        MatrixVectorFamily::GemvFast => EncodableVariant::GemvFast(
            <B::Kernels as Kernels>::QuantizedMatmulGemvFastV2Kernel::new(
                context,
                data_type,
                group_size,
                bits,
                use_zero_points,
                use_mlx_quant,
            )
            .map_err(QuantizedMatmulError::BackendError)?,
        ),
        MatrixVectorFamily::VectorMatrix => EncodableVariant::VectorMatrix(
            <B::Kernels as Kernels>::QuantizedMatmulVectorMatrixV2Kernel::new(
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
        MatrixMatrixFamily::GemmAlignedK => EncodableVariant::Gemm(
            <B::Kernels as Kernels>::QuantizedMatmulGemmV2Kernel::new(
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
        MatrixMatrixFamily::GemmUnalignedK => EncodableVariant::Gemm(
            <B::Kernels as Kernels>::QuantizedMatmulGemmV2Kernel::new(
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
        MatrixMatrixFamily::GemmTransposedAlignedN => EncodableVariant::GemmTransposed(
            <B::Kernels as Kernels>::QuantizedMatmulGemmTransposedV2Kernel::new(
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
        MatrixMatrixFamily::GemmTransposedUnalignedN => EncodableVariant::GemmTransposed(
            <B::Kernels as Kernels>::QuantizedMatmulGemmTransposedV2Kernel::new(
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
        MatrixMatrixFamily::GemmTransposed64x64 => EncodableVariant::GemmTransposed64x64(
            <B::Kernels as Kernels>::QuantizedMatmulGemmTransposed64x64V2Kernel::new(
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
    configuration: &QuantizedMatmulConfiguration,
) -> Result<(), QuantizedMatmulError<B>> {
    if !matches!(
        configuration.data_type,
        DataType::F16 | DataType::BF16 | DataType::F32
    ) {
        return Err(QuantizedMatmulError::UnsupportedDataType(
            configuration.data_type,
        ));
    }

    if !matches!(configuration.group_size, 32 | 64 | 128) {
        return Err(QuantizedMatmulError::UnsupportedGroupSize(
            configuration.group_size,
        ));
    }

    let _ = quant_bits(configuration.mode)?;
    Ok(())
}

fn select_matrix_vector_family(
    configuration: &QuantizedMatmulConfiguration,
) -> MatrixVectorFamily {
    if configuration.weights_transposed {
        if configuration.output_dim % 8 == 0 && configuration.input_dim % 512 == 0 {
            MatrixVectorFamily::GemvFast
        } else {
            MatrixVectorFamily::Gemv
        }
    } else {
        MatrixVectorFamily::VectorMatrix
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
            MatrixMatrixFamily::GemmTransposed64x64
        } else if aligned_n {
            MatrixMatrixFamily::GemmTransposedAlignedN
        } else {
            MatrixMatrixFamily::GemmTransposedUnalignedN
        }
    } else if configuration.input_dim % 32 == 0 {
        MatrixMatrixFamily::GemmAlignedK
    } else {
        MatrixMatrixFamily::GemmUnalignedK
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
    i32::try_from(value).map_err(|_| QuantizedMatmulError::ValueOutOfRange { name, value })
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

pub fn use_v2() -> bool {
    std::env::var("UZU_QUANT_MATMUL_V2")
        .map(|value| value == "1")
        .unwrap_or(false)
}

pub enum QuantizedMatmulKernelSwitchable<B: Backend> {
    V1(QuantizedMatmulKernelEncodable<B>),
    V2(QuantizedMatmulKernelEncodableV2<B>),
}

impl<B: Backend> QuantizedMatmulKernelSwitchable<B> {
    pub fn new(
        context: &B::Context,
        configuration: QuantizedMatmulConfiguration,
    ) -> Result<Self, QuantizedMatmulError<B>> {
        if use_v2() {
            QuantizedMatmulKernelEncodableV2::new(context, configuration).map(Self::V2)
        } else {
            QuantizedMatmulKernelEncodable::new(context, configuration).map(Self::V1)
        }
    }

    pub fn encode(
        &self,
        encoder: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
        arguments: QuantizedMatmulArguments<B>,
    ) -> Result<(), QuantizedMatmulError<B>> {
        match self {
            Self::V1(kernel) => kernel.encode(encoder, arguments),
            Self::V2(kernel) => kernel.encode(encoder, arguments),
        }
    }
}

fn kernel_key_name(key: KernelKey) -> &'static str {
    match key {
        KernelKey::MatrixVector(MatrixVectorFamily::Gemv) => "matrix_vector_gemv",
        KernelKey::MatrixVector(MatrixVectorFamily::GemvFast) => "matrix_vector_gemv_fast",
        KernelKey::MatrixVector(MatrixVectorFamily::VectorMatrix) => "matrix_vector_vector_matrix",
        KernelKey::MatrixMatrix(MatrixMatrixFamily::GemmAlignedK) => "matrix_matrix_gemm_aligned_k",
        KernelKey::MatrixMatrix(MatrixMatrixFamily::GemmUnalignedK) => {
            "matrix_matrix_gemm_unaligned_k"
        }
        KernelKey::MatrixMatrix(MatrixMatrixFamily::GemmTransposedAlignedN) => {
            "matrix_matrix_gemm_transposed_aligned_n"
        }
        KernelKey::MatrixMatrix(MatrixMatrixFamily::GemmTransposedUnalignedN) => {
            "matrix_matrix_gemm_transposed_unaligned_n"
        }
        KernelKey::MatrixMatrix(MatrixMatrixFamily::GemmTransposed64x64) => {
            "matrix_matrix_gemm_transposed_64x64"
        }
    }
}
