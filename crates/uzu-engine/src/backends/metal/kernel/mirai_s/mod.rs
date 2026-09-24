use super::{MiraiSNarrowProjectionMetalKernel, MiraiSProjectionMetalKernel, MiraiSTransformMetalKernel};
use crate::{
    backends::{
        common::{
            Allocation, Encoder,
            kernel::mirai_s::{MiraiSProjection, MiraiSTransform, ProjectionArguments, RotatedInput, TrellisCodec},
        },
        metal::{Metal, MetalContext, error::MetalError},
    },
    data_type::DataType,
};

const MAX_TOKEN_TILE: u32 = 64;

pub struct MetalMiraiSTransform {
    kernel: MiraiSTransformMetalKernel,
    columns: u32,
}

impl MiraiSTransform for MetalMiraiSTransform {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        columns: u32,
    ) -> Result<Option<Self>, MetalError> {
        if !context.supports_mxu || ![5120, 6144, 17408].contains(&columns) {
            return Ok(None);
        }
        Ok(Some(Self {
            kernel: MiraiSTransformMetalKernel::new(context, columns)?,
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
        // the projection's MXU reads whole token tiles; rows past `batch` are never written and never stored
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

/// The projection kernels of one codec.
pub struct MetalMiraiSProjection {
    wide_32: MiraiSProjectionMetalKernel,
    wide_64: MiraiSProjectionMetalKernel,
    narrow_2: MiraiSNarrowProjectionMetalKernel,
    narrow_4: MiraiSNarrowProjectionMetalKernel,
    busy_simdgroups: u32,
}

impl MiraiSProjection for MetalMiraiSProjection {
    type Backend = Metal;

    fn new(
        context: &MetalContext,
        codec: TrellisCodec,
    ) -> Result<Option<Self>, MetalError> {
        if !context.supports_mxu {
            return Ok(None);
        }
        let (vector_width, transition_bits) = (codec.vector_width(), codec.transition_bits());
        Ok(Some(Self {
            wide_32: MiraiSProjectionMetalKernel::new(context, 32, vector_width, transition_bits)?,
            wide_64: MiraiSProjectionMetalKernel::new(context, 64, vector_width, transition_bits)?,
            narrow_2: MiraiSNarrowProjectionMetalKernel::new(context, 2, vector_width, transition_bits)?,
            narrow_4: MiraiSNarrowProjectionMetalKernel::new(context, 4, vector_width, transition_bits)?,
            busy_simdgroups: match codec {
                TrellisCodec::Vector4Restart64 => 512,
                TrellisCodec::Vector2Transition6 | TrellisCodec::Vector2Transition4 => 256,
            },
        }))
    }

    fn encode(
        &self,
        arguments: ProjectionArguments<'_, Metal>,
        encoder: &mut Encoder<Metal>,
    ) {
        let ProjectionArguments {
            input,
            codes,
            row_scales,
            codebook,
            rows,
            output,
            output_row_offset,
            output_stride,
        } = arguments;
        assert!(output_row_offset + rows <= output_stride);
        let output = (output, output_row_offset as usize * DataType::BF16.size_in_bytes());
        macro_rules! encode {
            ($kernel:expr) => {
                $kernel.encode(
                    codes,
                    &input.activations,
                    &input.token_statistics,
                    row_scales,
                    codebook,
                    output,
                    rows,
                    input.columns,
                    input.batch,
                    output_stride,
                    encoder,
                )
            };
        }
        // The narrow kernels (16 tokens per MXU tile) win at batch <= 16 once their larger row tiles still leave
        // enough SIMDgroups to fill the GPU: 256 for V2, which reads its levels from a table, 512 for V4, which hashes
        // them. Measured on an M5 Pro per projection with codes streamed from DRAM, narrow vs wide: V2 34816 x 5120
        // 426 vs 534 us, 16480 x 5120 215 vs 277, 8192 x 5120 124 vs 148, 5120 x 17408 308 vs 256; V4 34816 x 5120
        // 371 vs 455, 16480 x 5120 177 vs 260, 6144 x 5120 132 vs 92.
        if input.batch <= 16 && rows / 64 >= self.busy_simdgroups {
            encode!(self.narrow_4)
        } else if input.batch <= 16 && rows / 32 >= self.busy_simdgroups {
            encode!(self.narrow_2)
        } else if input.batch <= 32 {
            encode!(self.wide_32)
        } else {
            encode!(self.wide_64)
        }
    }
}

#[cfg(test)]
#[path = "../../../../../unit/backends/metal/kernel/mirai_s_test.rs"]
mod tests;
