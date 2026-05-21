pub mod gemm;
pub mod gemv;

use std::sync::OnceLock;

use self::{
    gemm::GemmKernel,
    gemv::{GemvKernel, QuantGemvKernel},
};
use crate::{
    DataType,
    backends::{
        common::{
            Backend, Encoder,
            gpu_types::gemm::GemmDTransform,
            kernel::{
                HadamardTransformKernel, Kernels, TensorAddBiasKernel,
                matmul::{MatmulArguments, MatmulB, MatmulDOp, MatmulError, MatmulKernel},
            },
        },
        metal::{Metal, context::MetalContext, kernel::TensorAddBiasMetalKernel},
    },
};

pub struct MatmulMetalKernel {
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

#[derive(Debug, Clone, Copy)]
pub enum MatmulDispatchPath {
    Auto,
    Gemv,
    Gemm,
    QuantGemv,
    QuantGemm,
}

impl MatmulMetalKernel {
    pub fn encode_with_path<'a>(
        &mut self,
        arguments: MatmulArguments<'a, Metal>,
        encoder: &mut Encoder<Metal>,
        path: MatmulDispatchPath,
    ) -> Result<(), MatmulError<Metal>> {
        match (path, &arguments.b) {
            (MatmulDispatchPath::Auto, _) => self.encode(arguments, encoder),
            (
                MatmulDispatchPath::Gemv,
                MatmulB::FullPrecision {
                    ..
                },
            ) => self.dispatch_fp_gemv(arguments, encoder),
            (
                MatmulDispatchPath::Gemm,
                MatmulB::FullPrecision {
                    ..
                },
            ) => self.dispatch_fp_gemm(arguments, encoder),
            (
                MatmulDispatchPath::QuantGemv,
                MatmulB::ScaleBiasDequant {
                    ..
                }
                | MatmulB::ScaleZeroPointDequant {
                    ..
                },
            ) => self.dispatch_quant_gemv(arguments, encoder),
            (
                MatmulDispatchPath::QuantGemm,
                MatmulB::ScaleBiasDequant {
                    ..
                }
                | MatmulB::ScaleZeroPointDequant {
                    ..
                },
            ) => self.dispatch_quant_gemm(arguments, encoder),
            _ => panic!("MatmulDispatchPath does not match MatmulB variant"),
        }
    }

    fn dispatch_fp_gemv<'a>(
        &mut self,
        arguments: MatmulArguments<'a, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        // FP gemv handles SCALE/ACCUMULATE/BIAS natively; pull RHT out as post-pass.
        let post_rht = arguments.d_transform.iter().find_map(|op| op.as_rht());

        let MatmulArguments {
            a,
            a_offset,

            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            mut d_transform,
            m,
            n,
            k,
        } = arguments;
        d_transform.retain(|op| op.bit() != GemmDTransform::RHT);

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
            self.hadamard.encode(d, factors, n, m, encoder);
        }
        Ok(())
    }

    fn dispatch_fp_gemm<'a>(
        &mut self,
        arguments: MatmulArguments<'a, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        // FP gemm core handles SCALE/ACCUMULATE natively via SPECIALIZE.
        // Simdgroup path fuses BIAS too; MXU path still needs post-pass bias.
        // RHT always post-pass.
        let uses_mxu = self.gemm.uses_mxu();
        let post_bias = if uses_mxu {
            arguments.d_transform.iter().find_map(|op| op.as_bias())
        } else {
            None
        };
        let post_rht = arguments.d_transform.iter().find_map(|op| op.as_rht());

        let MatmulArguments {
            a,
            a_offset,

            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            mut d_transform,
            m,
            n,
            k,
        } = arguments;
        // Strip ops that get post-passed here so the inner kernel sees only the
        // ops it fuses. MXU strips BIAS too (it falls back to post-pass).
        d_transform.retain(|op| {
            let bit = op.bit();
            if bit == GemmDTransform::RHT {
                return false;
            }
            if uses_mxu && bit == GemmDTransform::BIAS {
                return false;
            }
            true
        });

        let context = encoder.context();
        self.gemm
            .encode(
                context,
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
                encoder,
            )
            .map_err(MatmulError::BackendError)?;

        if let Some(bias) = post_bias {
            self.bias_add.encode(None::<&<Metal as Backend>::DenseBuffer>, bias, &mut *d, n, m * n, encoder);
        }
        if let Some(factors) = post_rht {
            self.hadamard.encode(d, factors, n, m, encoder);
        }
        Ok(())
    }

    fn dispatch_quant_gemv<'a>(
        &mut self,
        arguments: MatmulArguments<'a, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        // Quant gemv handles RHT (fused via qmv_fast when eligible); pull BIAS
        // out as post-pass. SCALE/ACCUMULATE are rejected by the inner kernel.
        let post_bias = arguments.d_transform.iter().find_map(|op| op.as_bias());

        let MatmulArguments {
            a,
            a_offset,

            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            mut d_transform,
            m,
            n,
            k,
        } = arguments;
        d_transform.retain(|op| op.bit() != GemmDTransform::BIAS);

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
            self.bias_add.encode(None::<&<Metal as Backend>::DenseBuffer>, bias, d, n, m * n, encoder);
        }
        Ok(())
    }

    fn dispatch_quant_gemm<'a>(
        &mut self,
        arguments: MatmulArguments<'a, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let mask = MatmulDOp::mask(&arguments.d_transform);
        if mask.contains(GemmDTransform::ACCUMULATE) {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::ACCUMULATE,
                path: "QuantGemm",
            });
        }
        if !arguments.b_transpose || arguments.b_leading_dimension.is_some() {
            return Err(MatmulError::UnsupportedLayout {
                path: "QuantGemm",
            });
        }

        // Quant core handles SCALE via GemmParams.ab_scale and BIAS via the
        // fused output epilogue (simdgroup path). RHT is post-pass.
        let post_rht = arguments.d_transform.iter().find_map(|op| op.as_rht());

        let MatmulArguments {
            a,
            a_offset,

            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            mut d_transform,
            m,
            n,
            k,
        } = arguments;
        d_transform.retain(|op| op.bit() != GemmDTransform::RHT);

        let context = encoder.context();
        self.gemm
            .encode(
                context,
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
                encoder,
            )
            .map_err(MatmulError::BackendError)?;

        if let Some(factors) = post_rht {
            self.hadamard.encode(d, factors, n, m, encoder);
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
        let quant_gemv = QuantGemvKernel::new(context, data_type);
        let hadamard = <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(context, data_type)
            .map_err(MatmulError::BackendError)?;

        Ok(Self {
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
            MatmulB::FullPrecision {
                ..
            } => {
                let gemv_eligible = arguments.b_transpose
                    && arguments.b_offset == 0
                    && arguments.b_leading_dimension.is_none_or(|ld| ld == arguments.k)
                    && arguments.m <= max_gemv_batch_threshold();

                if gemv_eligible {
                    self.dispatch_fp_gemv(arguments, encoder)
                } else {
                    self.dispatch_fp_gemm(arguments, encoder)
                }
            },
            MatmulB::ScaleBiasDequant {
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                ..
            } => {
                if arguments.m >= 5 && arguments.n > 1 {
                    self.dispatch_quant_gemm(arguments, encoder)
                } else {
                    self.dispatch_quant_gemv(arguments, encoder)
                }
            },
        }
    }
}
