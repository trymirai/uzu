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
            Allocation, Backend, Encoder,
            gpu_types::gemm::{GemmAPrologue, GemmDTransform},
            kernel::{
                HadamardTransformKernel, Kernels, TensorAddBiasKernel,
                matmul::{
                    MatmulArguments, MatmulB, MatmulError, MatmulKernel, ResolvedAPrologue, ResolvedDTransform,
                    resolve_a, resolve_d,
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
    pub fn encode_with_path<'a>(
        &mut self,
        arguments: MatmulArguments<'a, Metal>,
        encoder: &mut Encoder<Metal>,
        path: MatmulDispatchPath,
    ) -> Result<(), MatmulError<Metal>> {
        match (path, &arguments.b) {
            (MatmulDispatchPath::Auto, _) => self.encode(arguments, encoder),
            (MatmulDispatchPath::Gemv, MatmulB::FullPrecision { .. }) => self.dispatch_fp_gemv(arguments, encoder),
            (MatmulDispatchPath::Gemm, MatmulB::FullPrecision { .. }) => {
                self.dispatch_fp_gemm(arguments, encoder, false)
            },
            (MatmulDispatchPath::GemmMxu, MatmulB::FullPrecision { .. }) => {
                self.dispatch_fp_gemm(arguments, encoder, true)
            },
            (
                MatmulDispatchPath::QuantGemv,
                MatmulB::ScaleBiasDequant { .. } | MatmulB::ScaleZeroPointDequant { .. },
            ) => self.dispatch_quant_gemv(arguments, encoder),
            (
                MatmulDispatchPath::QuantGemm,
                MatmulB::ScaleBiasDequant { .. } | MatmulB::ScaleZeroPointDequant { .. },
            ) => self.dispatch_quant_gemm(arguments, encoder),
            _ => panic!("MatmulDispatchPath does not match MatmulB variant"),
        }
    }

    fn validate_no_a_prologue(
        resolved_a: &ResolvedAPrologue<'_, Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        if resolved_a.mask.contains(GemmAPrologue::RHT) {
            return Err(MatmulError::UnsupportedAOp {
                bit: GemmAPrologue::RHT,
                path: "MatmulMetalKernel",
            });
        }
        Ok(())
    }

    fn apply_d_post_passes(
        &mut self,
        resolved_d: ResolvedDTransform<'_, Metal>,
        d: &mut Allocation<Metal>,
        m: u32,
        n: u32,
        encoder: &mut Encoder<Metal>,
    ) {
        if let Some(bias) = resolved_d.bias {
            self.bias_add.encode(
                None::<&<Metal as Backend>::DenseBuffer>,
                bias,
                &mut *d,
                n,
                m * n,
                encoder,
            );
        }
        if let Some(factors) = resolved_d.rht {
            self.hadamard.encode(d, factors, n, m, encoder);
        }
    }

    fn dispatch_fp_gemv<'a>(
        &mut self,
        arguments: MatmulArguments<'a, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let resolved_a = resolve_a(arguments.a_prologue)?;
        Self::validate_no_a_prologue(&resolved_a)?;
        let resolved_d = resolve_d(arguments.d_transform)?;

        let MatmulArguments {
            a,
            a_offset,
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            m,
            n,
            k,
            ..
        } = arguments;

        // Inner core handles SCALE/ACCUMULATE/BIAS natively; only RHT is post-pass.
        let inner_d = ResolvedDTransform {
            mask: resolved_d.mask,
            ab_scale: resolved_d.ab_scale,
            bias: resolved_d.bias,
            rht: None,
        };
        let synthetic = MatmulArguments {
            a,
            a_offset,
            a_prologue: &[],
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d: &mut *d,
            d_transform: &[],
            m,
            n,
            k,
        };
        gemv::fp::encode(&mut self.gemv, encoder, synthetic, inner_d)?;

        if let Some(factors) = resolved_d.rht {
            self.hadamard.encode(d, factors, n, m, encoder);
        }
        Ok(())
    }

    fn dispatch_fp_gemm<'a>(
        &mut self,
        arguments: MatmulArguments<'a, Metal>,
        encoder: &mut Encoder<Metal>,
        use_mxu: bool,
    ) -> Result<(), MatmulError<Metal>> {
        let resolved_a = resolve_a(arguments.a_prologue)?;
        Self::validate_no_a_prologue(&resolved_a)?;
        let resolved_d = resolve_d(arguments.d_transform)?;

        let MatmulArguments {
            a,
            a_offset,
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            m,
            n,
            k,
            ..
        } = arguments;

        // FP gemm core handles SCALE/ACCUMULATE natively via SPECIALIZE;
        // BIAS/RHT are outer post-pass.
        let inner_d = ResolvedDTransform {
            mask: resolved_d.mask & (GemmDTransform::SCALE | GemmDTransform::ACCUMULATE),
            ab_scale: resolved_d.ab_scale,
            bias: None,
            rht: None,
        };
        let inner_args = MatmulArguments {
            a,
            a_offset,
            a_prologue: &[],
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d: &mut *d,
            d_transform: &[],
            m,
            n,
            k,
        };
        let context = encoder.context();
        self.gemm
            .encode(context, encoder, inner_args, inner_d, use_mxu)
            .map_err(MatmulError::BackendError)?;

        // Post-passes: BIAS and RHT.
        let post = ResolvedDTransform {
            mask: resolved_d.mask & (GemmDTransform::BIAS | GemmDTransform::RHT),
            ab_scale: 1.0,
            bias: resolved_d.bias,
            rht: resolved_d.rht,
        };
        self.apply_d_post_passes(post, d, m, n, encoder);
        Ok(())
    }

    fn dispatch_quant_gemv<'a>(
        &mut self,
        arguments: MatmulArguments<'a, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let resolved_a = resolve_a(arguments.a_prologue)?;
        Self::validate_no_a_prologue(&resolved_a)?;
        let resolved_d = resolve_d(arguments.d_transform)?;

        let MatmulArguments {
            a,
            a_offset,
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            m,
            n,
            k,
            ..
        } = arguments;

        // Quant gemv handles RHT (fused via qmv_fast when eligible); BIAS is post-pass.
        // SCALE/ACCUMULATE are rejected by the kernel itself.
        let inner_d = ResolvedDTransform {
            mask: resolved_d.mask & (GemmDTransform::RHT),
            ab_scale: resolved_d.ab_scale,
            bias: None,
            rht: resolved_d.rht,
        };
        let inner_args = MatmulArguments {
            a,
            a_offset,
            a_prologue: &[],
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d: &mut *d,
            d_transform: &[],
            m,
            n,
            k,
        };
        self.quant_gemv.encode(encoder, inner_args, inner_d)?;

        if let Some(bias) = resolved_d.bias {
            self.bias_add.encode(
                None::<&<Metal as Backend>::DenseBuffer>,
                bias,
                d,
                n,
                m * n,
                encoder,
            );
        }
        Ok(())
    }

    fn dispatch_quant_gemm<'a>(
        &mut self,
        arguments: MatmulArguments<'a, Metal>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MatmulError<Metal>> {
        let resolved_a = resolve_a(arguments.a_prologue)?;
        Self::validate_no_a_prologue(&resolved_a)?;
        let resolved_d = resolve_d(arguments.d_transform)?;

        if resolved_d.mask.contains(GemmDTransform::ACCUMULATE) {
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

        let MatmulArguments {
            a,
            a_offset,
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d,
            m,
            n,
            k,
            ..
        } = arguments;

        // Quant core handles SCALE via GemmParams.ab_scale; BIAS/RHT are post-pass.
        let inner_d = ResolvedDTransform {
            mask: resolved_d.mask & GemmDTransform::SCALE,
            ab_scale: resolved_d.ab_scale,
            bias: None,
            rht: None,
        };
        let inner_args = MatmulArguments {
            a,
            a_offset,
            a_prologue: &[],
            b,
            b_offset,
            b_leading_dimension,
            b_transpose,
            d: &mut *d,
            d_transform: &[],
            m,
            n,
            k,
        };
        let context = encoder.context();
        self.gemm
            .encode(context, encoder, inner_args, inner_d, false)
            .map_err(MatmulError::BackendError)?;

        // Post-passes: BIAS and RHT.
        if let Some(bias) = resolved_d.bias {
            self.bias_add.encode(
                None::<&<Metal as Backend>::DenseBuffer>,
                bias,
                &mut *d,
                n,
                m * n,
                encoder,
            );
        }
        if let Some(factors) = resolved_d.rht {
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
            MatmulB::FullPrecision { .. } => {
                let context = encoder.context();
                let gemv_eligible = arguments.b_transpose
                    && arguments.b_offset == 0
                    && arguments.b_leading_dimension.is_none_or(|ld| ld == arguments.k)
                    && arguments.m <= max_gemv_batch_threshold();

                if gemv_eligible {
                    self.dispatch_fp_gemv(arguments, encoder)
                } else {
                    let use_mxu = self.is_mxu_eligible(context);
                    self.dispatch_fp_gemm(arguments, encoder, use_mxu)
                }
            },
            MatmulB::ScaleBiasDequant { .. } | MatmulB::ScaleZeroPointDequant { .. } => {
                if arguments.m >= 5 && arguments.n > 1 {
                    self.dispatch_quant_gemm(arguments, encoder)
                } else {
                    self.dispatch_quant_gemv(arguments, encoder)
                }
            },
        }
    }
}
