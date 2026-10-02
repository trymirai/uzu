use crate::backends::common::{Backend, BufferMut, BufferRef, CommandBuffer, Kernels, kernel::TrellisTransformKernel};

const SUPPORTED_COLUMNS: [u32; 3] = [5120, 6144, 17408];

/// Mixing matrix side length: columns divided by their largest power-of-two factor
pub fn mixing_dimension(columns: u32) -> u32 {
    columns >> columns.trailing_zeros()
}

pub struct TrellisTransform<B: Backend> {
    kernel: <B::Kernels as Kernels>::TrellisTransformKernel,
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
        }))
    }

    pub fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        rht_factors: impl BufferRef<Backend = B>,
        mixing: impl BufferRef<Backend = B>,
        activations: impl BufferMut<Backend = B>,
        column_group_sums: impl BufferMut<Backend = B>,
        scales: impl BufferMut<Backend = B>,
        batch: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        self.kernel.encode(input, rht_factors, mixing, activations, column_group_sums, scales, batch, command_buffer);
    }
}
