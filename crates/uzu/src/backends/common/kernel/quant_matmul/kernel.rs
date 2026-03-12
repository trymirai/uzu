use std::collections::HashMap;

use crate::backends::common::{
    Backend, CommandBuffer,
    kernel::{
        QuantizedMatmulGemmKernel, QuantizedMatmulGemmTransposed64x64Kernel,
        QuantizedMatmulGemmTransposedKernel, QuantizedMatmulGemvFastKernel,
        QuantizedMatmulGemvKernel, QuantizedMatmulVectorMatrixKernel,
    },
};

use super::{
    QuantizedMatmulError, QuantizedMatmulType,
    arguments::QuantizedMatmulArguments,
    configuration::QuantizedMatmulConfiguration,
    variant::{EncodableVariant, KernelKey, RuntimeVariant},
};

pub struct QuantizedMatmulKernelEncodable<B: Backend> {
    kernels: HashMap<KernelKey, EncodableVariant<B>>,
    matrix_vector_key: KernelKey,
    matrix_matrix_key: KernelKey,
    output_dim: usize,
    quantization_type: QuantizedMatmulType,
}

impl<B: Backend> QuantizedMatmulKernelEncodable<B> {
    pub fn new(
        context: &B::Context,
        configuration: QuantizedMatmulConfiguration,
    ) -> Result<Self, QuantizedMatmulError<B>> {
        configuration.validate()?;

        let bits = configuration.mode.bits();
        let use_mlx_quant = matches!(configuration.quantization_type, QuantizedMatmulType::Mlx);

        let matrix_vector_family = configuration.matrix_vector_family();
        let matrix_matrix_family = configuration.matrix_matrix_family(bits);
        let matrix_vector_key = KernelKey::MatrixVector(matrix_vector_family);
        let matrix_matrix_key = KernelKey::MatrixMatrix(matrix_matrix_family);

        let group_size = to_i32::<B>("group_size", configuration.group_size)?;
        let bits = to_i32::<B>("bits", bits)?;

        let mut kernels = HashMap::new();
        kernels.insert(
            matrix_vector_key,
            matrix_vector_family.create_kernel::<B>(
                context,
                configuration.data_type,
                group_size,
                bits,
                use_mlx_quant,
            )?,
        );
        kernels.insert(
            matrix_matrix_key,
            matrix_matrix_family.create_kernel::<B>(
                context,
                configuration.data_type,
                group_size,
                bits,
                use_mlx_quant,
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
            .ok_or_else(|| QuantizedMatmulError::MissingKernel(key.to_string().into_boxed_str()))?;
        let k = to_i32::<B>("input_dim", arguments.input_dim)?;
        let n = to_i32::<B>("output_dim", arguments.output_dim)?;
        let m = to_i32::<B>("batch", arguments.batch)?;
        let (zero_points, biases) = self
            .quantization_type
            .split_buffers::<B>(arguments.zero_points_or_biases_buffer);
        let a_with_offset = (arguments.a_buffer, arguments.a_offset);

        match kernel {
            EncodableVariant::Gemv(kernel) => {
                kernel.encode(
                    arguments.b_buffer, arguments.scales_buffer,
                    zero_points, biases, a_with_offset, arguments.output_buffer,
                    k, n, m, encoder,
                );
            }
            EncodableVariant::GemvFast(kernel) => {
                kernel.encode(
                    arguments.b_buffer, arguments.scales_buffer,
                    zero_points, biases, a_with_offset, arguments.output_buffer,
                    k, n, m, encoder,
                );
            }
            EncodableVariant::VectorMatrix(kernel) => {
                kernel.encode(
                    arguments.b_buffer, arguments.scales_buffer,
                    zero_points, biases, a_with_offset, arguments.output_buffer,
                    k, n, m, encoder,
                );
            }
            EncodableVariant::Gemm(kernel) => {
                kernel.encode(
                    arguments.b_buffer, arguments.scales_buffer,
                    zero_points, biases, a_with_offset, arguments.output_buffer,
                    k, n, m, encoder,
                );
            }
            EncodableVariant::GemmTransposed(kernel) => {
                kernel.encode(
                    arguments.b_buffer, arguments.scales_buffer,
                    zero_points, biases, a_with_offset, arguments.output_buffer,
                    k, n, m, encoder,
                );
            }
            EncodableVariant::GemmTransposed64x64(kernel) => {
                kernel.encode(
                    arguments.b_buffer, arguments.scales_buffer,
                    zero_points, biases, a_with_offset, arguments.output_buffer,
                    k, n, m, encoder,
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

fn to_i32<B: Backend>(
    name: &'static str,
    value: usize,
) -> Result<i32, QuantizedMatmulError<B>> {
    i32::try_from(value).map_err(|_| QuantizedMatmulError::ValueOutOfRange { name, value })
}
