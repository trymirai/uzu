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
                matmul::{
                    MatmulArguments, MatmulError, MatmulKernel, MatmulWeights,
                },
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
    QuantGemv,
    QuantGemm,
}

impl MatmulMetalKernel {
    pub fn encode_with_path(
        &mut self,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<Metal>,
        path: MatmulDispatchPath,
    ) -> Result<(), MatmulError<Metal>> {
        match (path, &arguments.b) {
            (MatmulDispatchPath::Auto, _) => self.encode(arguments, encoder),
            (MatmulDispatchPath::Gemv, MatmulWeights::FullPrecision { .. }) => {
                self.encode_fp_gemv(arguments, encoder);
                Ok(())
            },
            (MatmulDispatchPath::Gemm, MatmulWeights::FullPrecision { .. }) => {
                self.encode_fp_gemm(arguments, encoder, false);
                Ok(())
            },
            (MatmulDispatchPath::GemmMxu, MatmulWeights::FullPrecision { .. }) => {
                self.encode_fp_gemm(arguments, encoder, true);
                Ok(())
            },
            (MatmulDispatchPath::QuantGemv, MatmulWeights::Quantized { .. }) => {
                self.encode_quant_gemv(arguments, encoder)
            },
            (MatmulDispatchPath::QuantGemm, MatmulWeights::Quantized { .. }) => {
                self.encode_quant_gemm(arguments, encoder)
            },
            _ => panic!("MatmulDispatchPath does not match MatmulWeights variant"),
        }
    }

    fn encode_fp_gemv(
        &mut self,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<Metal>,
    ) {
        gemv::fp::encode(&mut self.gemv, encoder, arguments).expect("Failed to encode GEMV");
    }

    fn encode_fp_gemm(
        &mut self,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<Metal>,
        use_mxu: bool,
    ) {
        let context = encoder.context();
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

    fn encode_quant_gemm(
        &mut self,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let MatmulArguments {
            a,
            a_offset,
            b,
            d,
            batch_dim,
            input_dim,
            output_dim,
        } = arguments;
        let MatmulWeights::Quantized {
            b: weights,
            scales,
            zero_points_or_biases,
            method,
            mode,
            group_size,
            hadamard_factors,
        } = b
        else {
            unreachable!("encode_quant_gemm requires Quantized weights");
        };

        let context = encoder.context();
        self.gemm
            .encode(
                context,
                encoder,
                GemmRequest::Quant {
                    method,
                    mode,
                    group_size,
                    a,
                    a_offset,
                    b: weights,
                    scales,
                    zero_points_or_biases,
                    d: &mut *d,
                    batch_dim,
                    input_dim,
                    output_dim,
                },
            )
            .map_err(MatmulError::BackendError)?;

        if let Some(factors) = hadamard_factors {
            self.hadamard.encode(d, factors, output_dim, batch_dim, encoder);
        }
        Ok(())
    }

    fn encode_quant_gemv(
        &mut self,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        self.quant_gemv.encode(encoder, arguments)
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
        let quant_gemv = QuantGemvKernel::new(context, data_type);
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
    ) -> Result<(), MatmulError<Metal>> {
        match &arguments.b {
            MatmulWeights::FullPrecision {
                b_transpose,
                b_offset,
                b_leading_dimension,
                ..
            } => {
                let context = encoder.context();
                let gemv_eligible = *b_transpose
                    && *b_offset == 0
                    && b_leading_dimension.is_none_or(|ld| ld == arguments.input_dim)
                    && arguments.batch_dim <= max_gemv_batch_threshold();

                if gemv_eligible {
                    self.encode_fp_gemv(arguments, encoder);
                } else {
                    let use_mxu = self.is_mxu_eligible(context);
                    self.encode_fp_gemm(arguments, encoder, use_mxu);
                }
                Ok(())
            },
            MatmulWeights::Quantized { .. } => {
                if arguments.batch_dim >= 5 && arguments.output_dim > 1 {
                    self.encode_quant_gemm(arguments, encoder)
                } else {
                    self.encode_quant_gemv(arguments, encoder)
                }
            },
        }
    }
}
