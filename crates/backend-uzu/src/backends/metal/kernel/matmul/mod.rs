pub mod gemm;
pub mod gemv;

use std::sync::OnceLock;

use self::gemv::GemvKernel;
pub use self::{
    gemm::{GemmDispatchPath, GemmKernel},
    gemv::QuantGemvKernel,
};
use crate::{
    backends::{
        common::{
            AsBufferRangeRef, Backend, Buffer, Encoder,
            gpu_types::gemm::GemmDTransform,
            kernel::{
                HadamardTransformKernel, TensorAddBiasKernel,
                matmul::{MatmulArguments, MatmulB, MatmulError, MatmulKernel, MatmulQuantCombo},
            },
        },
        metal::{Metal, context::MetalContext, error::MetalError},
    },
    data_type::DataType,
};

pub struct MatmulMetalKernel {
    gemv: GemvKernel,
    pub quant_gemv: QuantGemvKernel,
    pub gemm: GemmKernel,
}

const DEFAULT_GEMV_MAX_BATCH: u32 = 8;
static GEMV_MAX_BATCH: OnceLock<u32> = OnceLock::new();

fn max_gemv_batch_threshold() -> u32 {
    *GEMV_MAX_BATCH.get_or_init(|| {
        std::env::var("UZU_GEMV_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_GEMV_MAX_BATCH)
    })
}

impl MatmulMetalKernel {
    fn dispatch_gemv<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let post_rht = arguments.d_transform.rht_factors;

        let MatmulArguments {
            a,
            a_offset,
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            d_transform,
            m,
            n,
            k,
        } = arguments;

        // A temopary hack, will port the post ops to the gemv later.
        let d_transform = d_transform.without(GemmDTransform::RHT);

        gemv::fp::encode(
            &mut self.gemv,
            encoder,
            MatmulArguments {
                a,
                a_offset,
                b,
                b_offset,
                b_leading_dimension,
                b_transpose,
                d: &mut *d,
                d_transform,
                m,
                n,
                k,
            },
        )?;

        if let Some(factors) = post_rht {
            self.gemm.hadamard.encode(d, factors, n, m, encoder);
        }
        Ok(())
    }

    fn dispatch_quant_gemv<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        if arguments.b_offset != 0 {
            return Err(MatmulError::UnsupportedLayout {
                path: "QuantGemv",
            });
        }
        let post_bias = arguments.d_transform.bias;

        let MatmulArguments {
            a,
            a_offset,
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            d_transform,
            m,
            n,
            k,
        } = arguments;
        let d_transform = d_transform.without(GemmDTransform::BIAS);

        self.quant_gemv.encode(
            encoder,
            MatmulArguments {
                a,
                a_offset,
                b,
                b_offset,
                b_leading_dimension,
                b_transpose,
                d: &mut *d,
                d_transform,
                m,
                n,
                k,
            },
        )?;

        if let Some(bias) = post_bias {
            self.gemm.bias_add.encode(None::<&<Metal as Backend>::DenseBuffer>, bias, d, n, m * n, encoder);
        }
        Ok(())
    }
}

impl MatmulKernel for MatmulMetalKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
    ) -> Result<Self, MetalError> {
        for data_type in [weights_data_type, input_data_type, output_data_type] {
            if !matches!(data_type, DataType::BF16 | DataType::F32) {
                return Err(MatmulError::<Metal>::UnsupportedDataType(data_type).into());
            }
        }

        let gemm = GemmKernel::new(context, weights_data_type, input_data_type, output_data_type)?;
        let gemv = GemvKernel::new(context, weights_data_type, input_data_type, output_data_type)?;
        let quant_gemv = QuantGemvKernel::new(context, weights_data_type, input_data_type, output_data_type);

        Ok(Self {
            gemv,
            quant_gemv,
            gemm,
        })
    }

    fn encode<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        let is_quant = !matches!(arguments.b, MatmulB::FullPrecision { .. });
        let gemv_eligible = if is_quant {
            arguments.m < 5 || arguments.n == 1
        } else {
            arguments.b_transpose
                && arguments.b_offset == 0
                && arguments.b_leading_dimension.is_none_or(|ld| ld == arguments.k)
                && arguments.m <= max_gemv_batch_threshold()
        };

        if gemv_eligible {
            if is_quant {
                self.dispatch_quant_gemv(arguments, encoder)?;
            } else {
                self.dispatch_gemv(arguments, encoder)?;
            }
            Ok(())
        } else {
            self.gemm.encode(arguments, encoder)
        }
    }

    fn preheat_quant_combo(
        &mut self,
        context: &MetalContext,
        combo: MatmulQuantCombo,
    ) -> Result<(), MetalError> {
        self.gemm.preheat_quant_combo(context, combo)
    }
}
