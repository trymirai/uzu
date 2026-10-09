use crate::backends::common::{
    Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels, kernel::ContextRingUpdateKernel,
};

pub struct ContextRingUpdate<B: Backend> {
    kernel: <B::Kernels as Kernels>::ContextRingUpdateKernel,
}

impl<B: Backend> ContextRingUpdate<B> {
    pub fn new(context: &B::Context) -> Result<Self, B::Error> {
        Ok(Self {
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
        let mut command_buffer = command_buffer.span("update repetition penalty ring");
        self.kernel.encode(input, context_ring, suffix_repetition_length, input_length, &mut command_buffer);
    }
}
