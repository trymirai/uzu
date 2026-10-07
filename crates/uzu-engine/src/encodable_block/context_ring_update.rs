use crate::backends::common::{
    Backend, BlockName, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
    kernel::ContextRingUpdateKernel,
};

pub struct ContextRingUpdate<B: Backend> {
    name: BlockName,
    kernel: <B::Kernels as Kernels>::ContextRingUpdateKernel,
}

impl<B: Backend> ContextRingUpdate<B> {
    pub fn new(
        name: BlockName,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            name,
            kernel: <B::Kernels as Kernels>::ContextRingUpdateKernel::new(context)?,
        })
    }

    pub fn encode(
        &self,
        input: impl BufferRef<Backend = B>,
        context_ring: impl BufferMut<Backend = B>,
        suffix_repetition_length: u32,
        input_length: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        command_buffer.push_debug_group(&self.name);
        command_buffer.sample_start_timestamp(&self.name);
        self.kernel.encode(input, context_ring, suffix_repetition_length, input_length, command_buffer);
        command_buffer.sample_end_timestamp();
        command_buffer.pop_debug_group();
    }
}
