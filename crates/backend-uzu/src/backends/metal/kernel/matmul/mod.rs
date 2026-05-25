pub mod gemm;
pub mod gemv;

use std::sync::OnceLock;

pub use self::gemm::GemmDispatchPath;
use self::{
    gemm::GemmKernel,
    gemv::{GemvKernel, QuantGemvKernel},
};
use crate::{
    DataType,
    backends::{
        common::{
            AsBufferRangeRef, Backend, Buffer, Encoder,
            gpu_types::gemm::GemmDTransform,
            kernel::{
                HadamardTransformKernel, Kernels, TensorAddBiasKernel,
                matmul::{MatmulArguments, MatmulB, MatmulError, MatmulKernel},
            },
        },
        metal::{
            Metal, context::MetalContext, error::MetalError, kernel::TensorAddBiasMetalKernel,
            metal_extensions::DeviceExt,
        },
    },
};

pub struct MatmulMetalKernel {
    mxu_eligible: bool,
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
    Gemm(GemmDispatchPath),
    QuantGemv,
}

impl MatmulMetalKernel {
    pub fn encode_dispatch_path<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        encoder: &mut Encoder<Metal>,
        path: MatmulDispatchPath,
    ) -> Result<(), MetalError> {
        match (path, &arguments.b) {
            (MatmulDispatchPath::Auto, _) => self.dispatch_auto(arguments, encoder),
            (
                MatmulDispatchPath::Gemv,
                MatmulB::FullPrecision {
                    ..
                },
            ) => self.dispatch_fp_gemv(arguments, encoder).map_err(MetalError::from),
            (
                MatmulDispatchPath::Gemm(gemm_path),
                MatmulB::FullPrecision {
                    ..
                },
            ) => self.dispatch_fp_gemm(arguments, gemm_path, encoder).map_err(MetalError::from),
            (
                MatmulDispatchPath::Gemm(GemmDispatchPath::Simdgroup),
                MatmulB::ScaleBiasDequant {
                    ..
                }
                | MatmulB::ScaleZeroPointDequant {
                    ..
                },
            ) => self.dispatch_quant_gemm(arguments, encoder).map_err(MetalError::from),
            (
                MatmulDispatchPath::Gemm(GemmDispatchPath::Mxu),
                MatmulB::ScaleBiasDequant {
                    ..
                }
                | MatmulB::ScaleZeroPointDequant {
                    ..
                },
            ) => panic!("GemmDispatchPath::Mxu is not supported with quantized B"),
            (
                MatmulDispatchPath::QuantGemv,
                MatmulB::ScaleBiasDequant {
                    ..
                }
                | MatmulB::ScaleZeroPointDequant {
                    ..
                },
            ) => self.dispatch_quant_gemv(arguments, encoder).map_err(MetalError::from),
            _ => panic!("MatmulDispatchPath does not match MatmulB variant"),
        }
    }

    fn dispatch_auto<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        match &arguments.b {
            MatmulB::FullPrecision {
                ..
            } => {
                let gemv_eligible = arguments.b_transpose
                    && arguments.b_offset == 0
                    && arguments.b_leading_dimension.is_none_or(|ld| ld == arguments.k)
                    && arguments.m <= max_gemv_batch_threshold();

                if gemv_eligible {
                    self.dispatch_fp_gemv(arguments, encoder).map_err(MetalError::from)
                } else {
                    let gemm_path = if self.mxu_eligible {
                        GemmDispatchPath::Mxu
                    } else {
                        GemmDispatchPath::Simdgroup
                    };
                    self.dispatch_fp_gemm(arguments, gemm_path, encoder).map_err(MetalError::from)
                }
            },
            MatmulB::ScaleBiasDequant {
                ..
            }
            | MatmulB::ScaleZeroPointDequant {
                ..
            } => {
                if arguments.m >= 5 && arguments.n > 1 {
                    self.dispatch_quant_gemm(arguments, encoder).map_err(MetalError::from)
                } else {
                    self.dispatch_quant_gemv(arguments, encoder).map_err(MetalError::from)
                }
            },
        }
    }

    fn dispatch_fp_gemv<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
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
            self.hadamard.encode(d, factors, n, m, encoder);
        }
        Ok(())
    }

    fn dispatch_fp_gemm<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        gemm_path: GemmDispatchPath,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let uses_mxu = matches!(gemm_path, GemmDispatchPath::Mxu);
        let post_bias = if uses_mxu {
            arguments.d_transform.bias
        } else {
            None
        };
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
        let mut stripped = GemmDTransform::RHT;
        if uses_mxu {
            stripped |= GemmDTransform::BIAS;
        }
        let d_transform = d_transform.without(stripped);

        let context = encoder.context();
        self.gemm
            .encode_dispatch_path(
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
                gemm_path,
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
            self.bias_add.encode(None::<&<Metal as Backend>::DenseBuffer>, bias, d, n, m * n, encoder);
        }
        Ok(())
    }

    fn dispatch_quant_gemm<'a, TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<'a, Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        if arguments.d_transform.mask().contains(GemmDTransform::ACCUMULATE) {
            return Err(MatmulError::UnsupportedDOp {
                bit: GemmDTransform::ACCUMULATE,
                path: "QuantGemm",
            });
        }
        if !arguments.b_transpose || arguments.b_leading_dimension.is_some() || arguments.b_offset != 0 {
            return Err(MatmulError::UnsupportedLayout {
                path: "QuantGemm",
            });
        }

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
        let d_transform = d_transform.without(GemmDTransform::RHT);

        let context = encoder.context();
        self.gemm
            .encode_dispatch_path(
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
                GemmDispatchPath::Simdgroup,
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
    ) -> Result<Self, MetalError> {
        if !matches!(data_type, DataType::F16 | DataType::BF16) {
            return Err(MatmulError::<Metal>::UnsupportedDataType(data_type).into());
        }

        let mxu_eligible = context.device.supports_mxu() && matches!(data_type, DataType::F16 | DataType::BF16);
        let bias_add = TensorAddBiasMetalKernel::new(context, data_type, true)?;
        let gemm = GemmKernel::new(context, data_type)?;
        let gemv = GemvKernel::new(context, data_type).map_err(MetalError::from)?;
        let quant_gemv = QuantGemvKernel::new(context, data_type);
        let hadamard = <<Metal as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(context, data_type)?;

        Ok(Self {
            mxu_eligible,
            gemv,
            quant_gemv,
            gemm,
            bias_add,
            hadamard,
        })
    }

    fn encode<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        self.encode_dispatch_path(arguments, encoder, MatmulDispatchPath::Auto)
    }
}
