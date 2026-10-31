use crate::backends::common::{Backend, CommandBuffer, CommandBufferEncoding, Kernels, kernel::StateAdvanceKernel};

/// The accepted path of a verified tree, with the tree's keys, values, log decays and betas, along which the GDN state
/// has yet to advance: the next tree verification advances it on the way, any other reader of the state first.
pub struct AcceptedPath<B: Backend> {
    pub k: B::ScratchBuffer,
    pub v: B::ScratchBuffer,
    pub log_decay: B::ScratchBuffer,
    pub beta: B::ScratchBuffer,
    /// Root first, each node a child of the one before.
    pub indices: Box<[u32]>,
}

impl<B: Backend> AcceptedPath<B> {
    pub fn encode_advance(
        &self,
        state_advance: &<B::Kernels as Kernels>::StateAdvanceKernel,
        state: &mut B::GlobalGpuBuffer,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let indices = command_buffer.allocate_constant_from_slice(&self.indices)?;
        state_advance.encode(
            &self.k,
            &self.v,
            &self.log_decay,
            &self.beta,
            &indices,
            state,
            self.indices.len() as u32,
            command_buffer,
        );
        Ok(())
    }
}
