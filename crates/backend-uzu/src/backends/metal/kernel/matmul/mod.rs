pub mod gemm;
pub mod gemv;

use std::sync::OnceLock;

pub use self::gemm::{GemmDispatchPath, GemmKernel};
use self::gemv::GemvDispatch;
use crate::{
    backends::{
        common::{
            AsBufferRangeRef, Buffer, Encoder,
            kernel::matmul::{MatmulArguments, MatmulB, MatmulError, MatmulKernel, MatmulQuantCombo},
        },
        metal::{Metal, context::MetalContext, error::MetalError},
    },
    data_type::DataType,
};

pub struct MatmulMetalKernel {
    gemv: GemvDispatch,
    pub gemm: GemmKernel,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
}

const RESULTS_PER_SIMDGROUP: u32 = 4;

const DEFAULT_GEMV_MAX_BATCH: u32 = 8;
static GEMV_MAX_BATCH: OnceLock<u32> = OnceLock::new();

fn max_gemv_batch_threshold() -> u32 {
    *GEMV_MAX_BATCH.get_or_init(|| {
        std::env::var("UZU_GEMV_MAX_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_GEMV_MAX_BATCH)
    })
}

fn gemv_eligible<TB: AsBufferRangeRef>(
    args: &MatmulArguments<Metal, TB>,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
) -> bool {
    // Accumulation reads-modifies-writes the output; the partial-block clamp can make
    // two threadgroups touch the same row, so require exact 32-aligned coverage when
    // accumulating (no overlapping writes).
    if args.d_transform.accumulate && !args.n.is_multiple_of(32) {
        return false;
    }
    // Fused output RHT needs whole 32-wide Hadamard blocks.
    if args.d_transform.rht_factors.is_some() && !args.n.is_multiple_of(32) {
        return false;
    }
    let is_quant = !matches!(args.b, MatmulB::FullPrecision { .. });
    if is_quant {
        // The output-tail clamp handles partial row blocks, so any n >= 4 works.
        args.n >= RESULTS_PER_SIMDGROUP && args.m < 5
    } else {
        // No mixed-precision FP GEMV variant: F32 weights require F32 activations.
        let mixed_precision = weights_data_type == DataType::F32
            && (input_data_type != DataType::F32 || output_data_type != DataType::F32);
        !mixed_precision
            && args.n >= RESULTS_PER_SIMDGROUP
            && args.b_transpose
            && args.b_offset == 0
            && args.b_leading_dimension.is_none_or(|ld| ld == args.k)
            && args.m <= max_gemv_batch_threshold()
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
        let gemv = GemvDispatch::new(context, weights_data_type, input_data_type, output_data_type)
            .map_err(MetalError::from)?;

        Ok(Self {
            gemv,
            gemm,
            weights_data_type,
            input_data_type,
            output_data_type,
        })
    }

    fn encode<TB: AsBufferRangeRef<Buffer: Buffer<Backend = Metal>>>(
        &mut self,
        arguments: MatmulArguments<Metal, TB>,
        encoder: &mut Encoder<Metal>,
    ) -> Result<(), MetalError> {
        if gemv_eligible(&arguments, self.weights_data_type, self.input_data_type, self.output_data_type) {
            self.gemv.encode(arguments, encoder).map_err(MetalError::from)
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
