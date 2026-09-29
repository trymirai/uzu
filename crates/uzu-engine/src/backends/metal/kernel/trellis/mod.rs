use super::{
    TrellisNarrowProjectionMetalKernel, TrellisProjectionMetalKernel, TrellisSimdgroupProjectionMetalKernel,
    TrellisTransformMetalKernel,
};
use crate::{
    backends::{
        common::{
            Allocation, Encoder,
            kernel::trellis::{ProjectionArguments, RotatedInput, TrellisCodec, TrellisProjection, TrellisTransform},
        },
        metal::{Metal, MetalContext, error::MetalError},
    },
    data_type::DataType,
};

const MAX_TOKEN_TILE: u32 = 64;

pub struct MetalTrellisTransform {
    kernel: TrellisTransformMetalKernel,
    columns: u32,
}

impl TrellisTransform for MetalTrellisTransform {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        columns: u32,
    ) -> Result<Option<Self>, MetalError> {
        if ![5120, 6144, 17408].contains(&columns) {
            return Ok(None);
        }
        Ok(Some(Self {
            kernel: TrellisTransformMetalKernel::new(context, columns)?,
            columns,
        }))
    }

    fn encode(
        &self,
        input: &Allocation<Metal>,
        signs: &Allocation<Metal>,
        mixing: &Allocation<Metal>,
        batch: u32,
        encoder: &mut Encoder<Metal>,
    ) -> Result<RotatedInput<Metal>, MetalError> {
        // the projections read whole token tiles; rows past `batch` are never written and never stored
        let padded_batch = batch.next_multiple_of(MAX_TOKEN_TILE);
        let mut activations = encoder.allocate_scratch_for_shape(&[padded_batch, self.columns], DataType::I8)?;
        let mut token_statistics = encoder.allocate_scratch_for_shape(&[batch, 8], DataType::F32)?;
        self.kernel.encode(input, signs, mixing, &mut activations, &mut token_statistics, batch, encoder);
        Ok(RotatedInput {
            activations,
            token_statistics,
            batch,
            columns: self.columns,
        })
    }
}

/// The projection kernels of one codec: MXU int8 tensor ops on M5 and later, plain SIMDgroup arithmetic before.
pub enum MetalTrellisProjection {
    Mxu {
        wide_32: TrellisProjectionMetalKernel,
        wide_64: TrellisProjectionMetalKernel,
        narrow_2: TrellisNarrowProjectionMetalKernel,
        narrow_4: TrellisNarrowProjectionMetalKernel,
        busy_simdgroups: u32,
    },
    Simdgroup {
        tokens_1: TrellisSimdgroupProjectionMetalKernel,
        tokens_8: TrellisSimdgroupProjectionMetalKernel,
    },
}

impl TrellisProjection for MetalTrellisProjection {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        codec: TrellisCodec,
    ) -> Result<Option<Self>, MetalError> {
        let (vector_width, transition_bits) = codec.shape();
        if !context.supports_mxu {
            return Ok(Some(Self::Simdgroup {
                tokens_1: TrellisSimdgroupProjectionMetalKernel::new(context, 1, vector_width, transition_bits)?,
                tokens_8: TrellisSimdgroupProjectionMetalKernel::new(context, 8, vector_width, transition_bits)?,
            }));
        }
        Ok(Some(Self::Mxu {
            wide_32: TrellisProjectionMetalKernel::new(context, 32, vector_width, transition_bits)?,
            wide_64: TrellisProjectionMetalKernel::new(context, 64, vector_width, transition_bits)?,
            narrow_2: TrellisNarrowProjectionMetalKernel::new(context, 2, vector_width, transition_bits)?,
            narrow_4: TrellisNarrowProjectionMetalKernel::new(context, 4, vector_width, transition_bits)?,
            busy_simdgroups: 128 * vector_width,
        }))
    }

    fn encode(
        &self,
        arguments: ProjectionArguments<'_, Metal>,
        encoder: &mut Encoder<Metal>,
    ) {
        let args = arguments;
        assert!(args.output_row_offset + args.rows <= args.output_stride);
        let output = (args.output, args.output_row_offset as usize * DataType::BF16.size_in_bytes());
        let (rows, batch) = (args.rows, args.input.batch);
        macro_rules! encode {
            ($kernel:expr) => {
                $kernel.encode(
                    args.codes,
                    &args.input.activations,
                    &args.input.token_statistics,
                    args.row_scales,
                    args.codebook,
                    output,
                    rows,
                    args.input.columns,
                    batch,
                    args.output_stride,
                    encoder,
                )
            };
        }
        match self {
            Self::Simdgroup {
                tokens_1,
                tokens_8,
            } => {
                if batch == 1 {
                    encode!(tokens_1)
                } else {
                    encode!(tokens_8)
                }
            },
            // at batch <= 16 the narrow kernels win once their larger row tiles still fill the GPU (M5 Pro)
            Self::Mxu {
                wide_32,
                wide_64,
                narrow_2,
                narrow_4,
                busy_simdgroups,
            } => {
                if batch <= 16 && rows / 64 >= *busy_simdgroups {
                    encode!(narrow_4)
                } else if batch <= 16 && rows / 32 >= *busy_simdgroups {
                    encode!(narrow_2)
                } else if batch <= 32 {
                    encode!(wide_32)
                } else {
                    encode!(wide_64)
                }
            },
        }
    }
}

#[cfg(test)]
#[path = "../../../../../unit/backends/metal/kernel/trellis_test.rs"]
mod tests;
