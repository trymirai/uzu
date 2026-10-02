use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels, gpu_types::trellis,
        kernel::TrellisTransformKernel,
    },
    data_type::DataType,
};

const SUPPORTED_COLUMNS: [u32; 3] = [5120, 6144, 17408];

/// Mixing matrix side length: columns divided by their largest power-of-two factor
pub fn mixing_dimension(columns: u32) -> u32 {
    columns >> columns.trailing_zeros()
}

pub struct RotatedInput<B: Backend> {
    /// i8 `[batch, columns]`
    pub activations: B::ScratchBuffer,
    /// f32 `[batch, 4]`
    pub column_group_sums: B::ScratchBuffer,
    /// f32 `[batch]`
    pub scales: B::ScratchBuffer,
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
        // TODO: a columns agnostic kernel?
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
        rht_factors: impl BufferRef<Backend = B>,
        mixing: impl BufferRef<Backend = B>,
        batch: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<RotatedInput<B>, B::Error> {
        let mut activations = command_buffer.allocate_scratch_for_shape(&[batch, self.columns], DataType::I8)?;
        let mut column_group_sums =
            command_buffer.allocate_scratch_for_shape(&[batch, trellis::COLUMN_GROUP_COUNT], DataType::F32)?;
        let mut scales = command_buffer.allocate_scratch_for_shape(&[batch], DataType::F32)?;
        self.kernel.encode(
            input,
            rht_factors,
            mixing,
            &mut activations,
            &mut column_group_sums,
            &mut scales,
            batch,
            command_buffer,
        );
        Ok(RotatedInput {
            activations,
            column_group_sums,
            scales,
            batch,
            columns: self.columns,
        })
    }
}
