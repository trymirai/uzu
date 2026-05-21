pub mod gemm;
pub mod gemv;

use std::sync::OnceLock;

use self::{
    gemm::{GemmKernel, GemmRequest},
    gemv::{GemvKernel, QuantGemvKernel},
};
use crate::{
    DataType,
    backends::{
        common::{
            Backend, Encoder,
            kernel::{
                HadamardTransformKernel, Kernels, TensorAddBiasKernel,
                matmul::{MatmulArguments, MatmulError, MatmulKernel},
                quant_matmul::{QuantizedMatmulArguments, QuantizedMatmulConfiguration, QuantizedMatmulError},
            },
        },
        metal::{Metal, context::MetalContext, kernel::TensorAddBiasMetalKernel, metal_extensions::DeviceExt},
    },
};

pub struct MatmulMetalKernel {
    data_type: DataType,
    gemv: GemvKernel,
    quant_gemv: QuantGemvKernel,
    pub(crate) gemm: GemmKernel,
    pub(crate) bias_add: TensorAddBiasMetalKernel,
    hadamard: <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel,
}

const DEFAULT_GEMV_MAX_BATCH: u32 = 8;
static GEMV_MAX_BATCH: OnceLock<u32> = OnceLock::new();

fn max_gemv_batch_threshold() -> u32 {
    *GEMV_MAX_BATCH.get_or_init(|| {
        std::env::var("UZU_GEMV_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_GEMV_MAX_BATCH)
    })
}

impl MatmulMetalKernel {
    fn is_mxu_eligible(
        &self,
        context: &MetalContext,
    ) -> bool {
        context.device.supports_mxu() && matches!(self.data_type, DataType::F16 | DataType::BF16)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MatmulDispatchPath {
    Auto,
    Gemv,
    Gemm,
    GemmMxu,
}

impl MatmulMetalKernel {
    pub fn encode_with_path(
        &mut self,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<Metal>,
        path: MatmulDispatchPath,
    ) {
        let context = encoder.context();
        match path {
            MatmulDispatchPath::Auto => self.encode(arguments, encoder),
            MatmulDispatchPath::Gemv => {
                gemv::fp::encode(&mut self.gemv, encoder, arguments).expect("Failed to encode GEMV")
            },
            MatmulDispatchPath::Gemm => self
                .gemm
                .encode(
                    context,
                    encoder,
                    GemmRequest::Fp {
                        bias_add: &mut self.bias_add,
                        arguments,
                        use_mxu: false,
                    },
                )
                .expect("Failed to encode Gemm"),
            MatmulDispatchPath::GemmMxu => self
                .gemm
                .encode(
                    context,
                    encoder,
                    GemmRequest::Fp {
                        bias_add: &mut self.bias_add,
                        arguments,
                        use_mxu: true,
                    },
                )
                .expect("Failed to encode GemmMxu"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum QuantizedMatmulDispatchPath {
    Auto,
    Gemm,
}

impl MatmulMetalKernel {
    pub fn encode_quantized_with_path(
        &mut self,
        arguments: QuantizedMatmulArguments<Metal>,
        configuration: &QuantizedMatmulConfiguration,
        encoder: &mut Encoder<Metal>,
        path: QuantizedMatmulDispatchPath,
    ) -> Result<(), QuantizedMatmulError<Metal>> {
        match path {
            QuantizedMatmulDispatchPath::Auto => self.encode_quantized(arguments, configuration, encoder),
            QuantizedMatmulDispatchPath::Gemm => self.encode_quantized_gemm(arguments, configuration, encoder),
        }
    }

    fn encode_quantized_gemm(
        &mut self,
        arguments: QuantizedMatmulArguments<Metal>,
        configuration: &QuantizedMatmulConfiguration,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), QuantizedMatmulError<Metal>> {
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

        let context = encoder.context();
        self.gemm
            .encode(
                context,
                encoder,
                GemmRequest::Quant {
                    configuration,
                    arguments: QuantizedMatmulArguments {
                        a,
                        a_offset,
                        b,
                        scales,
                        zero_points_or_biases,
                        output: &mut *output,
                        hadamard_factors: None,
                        batch_dim,
                    },
                },
            )
            .map_err(QuantizedMatmulError::BackendError)?;

        if configuration.use_hadamard
            && let Some(factors) = hadamard_factors
        {
            self.hadamard.encode(output, factors, configuration.output_dim as u32, batch_dim as u32, encoder);
        }
        Ok(())
    }
}

impl MatmulKernel for MatmulMetalKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        if !matches!(data_type, DataType::F16 | DataType::BF16) {
            return Err(MatmulError::UnsupportedDataType(data_type));
        }

        let bias_add = TensorAddBiasMetalKernel::new(context, data_type, true).map_err(MatmulError::BackendError)?;
        let gemm = GemmKernel::new(context, data_type).map_err(MatmulError::BackendError)?;
        let gemv = GemvKernel::new(context, data_type)?;
        let quant_gemv = QuantGemvKernel::new(context, data_type).map_err(|e| match e {
            QuantizedMatmulError::BackendError(err) => MatmulError::BackendError(err),
            QuantizedMatmulError::UnsupportedDataType(dt) => MatmulError::UnsupportedDataType(dt),
            _ => unreachable!(),
        })?;
        let hadamard = <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(context, data_type)
            .map_err(MatmulError::BackendError)?;

        Ok(Self {
            data_type,
            gemv,
            quant_gemv,
            gemm,
            bias_add,
            hadamard,
        })
    }

    fn encode(
        &mut self,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<Metal>,
    ) {
        let context = encoder.context();
        let gemv_eligible = arguments.b_transpose
            && arguments.b_offset == 0
            && arguments.b_leading_dimension.is_none_or(|ld| ld == arguments.input_dim)
            && arguments.batch_dim <= max_gemv_batch_threshold();

        if gemv_eligible {
            gemv::fp::encode(&mut self.gemv, encoder, arguments).expect("Failed to encode GEMV kernel");
            return;
        }

        let use_mxu = self.is_mxu_eligible(context);
        self.gemm
            .encode(
                context,
                encoder,
                GemmRequest::Fp {
                    bias_add: &mut self.bias_add,
                    arguments,
                    use_mxu,
                },
            )
            .expect("Failed to encode GEMM");
    }

    fn encode_quantized(
        &mut self,
        arguments: QuantizedMatmulArguments<Metal>,
        configuration: &QuantizedMatmulConfiguration,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), QuantizedMatmulError<Metal>> {
        if arguments.batch_dim >= 5 && configuration.output_dim > 1 {
            self.encode_quantized_gemm(arguments, configuration, encoder)
        } else {
            self.quant_gemv.encode(encoder, arguments, configuration)
        }
    }
}
