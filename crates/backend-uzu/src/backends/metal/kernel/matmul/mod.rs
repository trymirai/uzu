pub mod gemm;
pub mod gemv;
pub mod quant;

use std::sync::OnceLock;

use self::{gemm::GemmKernel, gemv::GemvKernel};
use crate::{
    DataType,
    backends::{
        common::{
            Encoder,
            gpu_types::gemm::GemmComputeKind,
            kernel::{
                TensorAddBiasKernel,
                matmul::{MatmulArguments, MatmulError, MatmulKernel},
            },
        },
        metal::{Metal, context::MetalContext, kernel::TensorAddBiasMetalKernel, metal_extensions::DeviceExt},
    },
};

pub struct MatmulMetalKernel {
    data_type: DataType,
    gemv: GemvKernel,
    pub(crate) gemm: GemmKernel,
    pub(crate) bias_add: TensorAddBiasMetalKernel,
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

/// Explicit dispatch paths for testing individual kernels independent of production routing.
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
            MatmulDispatchPath::Gemm => gemm::fp::encode(
                &mut self.gemm,
                &mut self.bias_add,
                self.data_type,
                context,
                encoder,
                arguments,
                GemmComputeKind::SimdgroupMma,
            )
            .expect("Failed to encode Gemm"),
            MatmulDispatchPath::GemmMxu => gemm::fp::encode(
                &mut self.gemm,
                &mut self.bias_add,
                self.data_type,
                context,
                encoder,
                arguments,
                GemmComputeKind::MxuMma,
            )
            .expect("Failed to encode GemmMxu"),
        }
    }
}

impl MatmulKernel for MatmulMetalKernel {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        data_type: DataType,
    ) -> Result<Self, MatmulError<Metal>> {
        if !matches!(data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(MatmulError::UnsupportedDataType(data_type));
        }

        let bias_add = TensorAddBiasMetalKernel::new(context, data_type, true).map_err(MatmulError::BackendError)?;
        let gemm = GemmKernel::new(context, data_type).map_err(MatmulError::BackendError)?;
        let gemv = GemvKernel::new(context, data_type)?;

        Ok(Self {
            data_type,
            gemv,
            gemm,
            bias_add,
        })
    }

    fn encode(
        &mut self,
        arguments: MatmulArguments<Metal>,
        encoder: &mut Encoder<Metal>,
    ) {
        // Routing notes:
        // - gemv only supports the canonical B layout (transposed, contiguous, no
        //   offset); non-canonical inputs skip it.
        // - GEMM (both SimdgroupMma and MxuMma) templates on TRANSPOSE_WEIGHTS and
        //   threads `b_offset` / `b_leading_dimension` end-to-end, so either compute
        //   path is correct for any layout. We prefer MXU when the device supports it.
        let context = encoder.context();
        let gemv_eligible = arguments.b_transpose
            && arguments.b_offset == 0
            && arguments.b_leading_dimension.is_none_or(|ld| ld == arguments.input_dim)
            && arguments.batch_dim <= max_gemv_batch_threshold();

        if gemv_eligible {
            gemv::fp::encode(&mut self.gemv, encoder, arguments).expect("Failed to encode GEMV kernel");
            return;
        }

        let compute = if self.is_mxu_eligible(context) {
            GemmComputeKind::MxuMma
        } else {
            GemmComputeKind::SimdgroupMma
        };
        gemm::fp::encode(&mut self.gemm, &mut self.bias_add, self.data_type, context, encoder, arguments, compute)
            .expect("Failed to encode GEMM");
    }
}
