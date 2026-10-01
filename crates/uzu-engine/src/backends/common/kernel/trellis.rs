use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels, gpu_types::trellis,
        kernel::TrellisTransformKernel,
    },
    data_type::DataType,
};

const SUPPORTED_COLUMNS: [u32; 3] = [5120, 6144, 17408];

/// `columns` with all factors of two removed, e.g. 17408 = 17 × 1024 → 17.
pub fn mixing_order(columns: u32) -> u32 {
    columns >> columns.trailing_zeros()
}

pub struct RotatedInput<B: Backend> {
    /// i8 `[batch, columns]`
    pub activations: B::ScratchBuffer,
    /// f32 `[batch, class sums.., scale, 0, 0, 0]`
    pub token_statistics: B::ScratchBuffer,
    pub batch: u32,
    pub columns: u32,
}

pub struct TrellisTransform<B: Backend> {
    kernel: <B::Kernels as Kernels>::TrellisTransformKernel,
    columns: u32,
}

impl<B: Backend> TrellisTransform<B> {
    pub fn new(
        context: &B::Context,
        columns: u32,
    ) -> Result<Option<Self>, B::Error> {
        if !SUPPORTED_COLUMNS.contains(&columns) {
            return Ok(None);
        }
        let kernel = <B::Kernels as Kernels>::TrellisTransformKernel::new(context, columns)?;
        Ok(Some(Self {
            kernel,
            columns,
        }))
    }

    pub fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        signs: impl BufferRef<Backend = B>,
        mixing: impl BufferRef<Backend = B>,
        batch: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<RotatedInput<B>, B::Error> {
        let mut activations = command_buffer.allocate_scratch_for_shape(&[batch, self.columns], DataType::I8)?;
        let mut token_statistics =
            command_buffer.allocate_scratch_for_shape(&[batch, trellis::TOKEN_STATISTICS_LEN], DataType::F32)?;
        self.kernel.encode(input, signs, mixing, &mut activations, &mut token_statistics, batch, command_buffer);
        Ok(RotatedInput {
            activations,
            token_statistics,
            batch,
            columns: self.columns,
        })
    }
}
