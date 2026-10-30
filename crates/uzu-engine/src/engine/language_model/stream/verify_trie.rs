use std::mem::size_of;

use crate::backends::common::{
    Backend, BufferRef, CommandBuffer, CommandBufferEncoding, Context, Kernels, gpu_types::trie::TrieNode,
    kernel::VerifyTrieKernel,
};

pub struct TrieVerification<B: Backend> {
    pub accepted_indices: B::GlobalGpuBuffer,
    pub accepted_input_tokens: B::GlobalGpuBuffer,
    pub accepted_output_tokens: B::GlobalGpuBuffer,
    pub accepted_output_norm: B::ScratchBuffer,
}

pub struct TrieVerifier<B: Backend> {
    kernel: <B::Kernels as Kernels>::VerifyTrieKernel,
}

impl<B: Backend> TrieVerifier<B> {
    pub fn new(context: &B::Context) -> Result<Self, B::Error> {
        Ok(Self {
            kernel: <B::Kernels as Kernels>::VerifyTrieKernel::new(context)?,
        })
    }

    pub fn encode(
        &self,
        trie_nodes: &[TrieNode],
        parents: &[i32],
        token_ids: impl BufferRef<Backend = B>,
        sampled_tokens: impl BufferRef<Backend = B>,
        output_norm: impl BufferRef<Backend = B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TrieVerification<B>, B::Error> {
        assert_eq!(trie_nodes.len(), parents.len(), "trie nodes and parents must be parallel");
        assert!(!trie_nodes.is_empty(), "the trie needs at least a root");

        let command_buffer = &mut *command_buffer.debug_group("verify trie");

        let num_nodes = trie_nodes.len() as u32;
        let max_accepted = trie_nodes.iter().map(|node| node.height).max().unwrap_or_default() + 1;
        let capacity = (max_accepted + 1) as usize * size_of::<u32>();
        let norm_row_bytes = output_norm.size() / trie_nodes.len();

        let trie_nodes = command_buffer.allocate_constant_from_slice(trie_nodes)?;
        let parents = command_buffer.allocate_constant_from_slice(parents)?;
        let mut accepted_indices = command_buffer.context().create_buffer_gpu(capacity)?;
        let mut accepted_input_tokens = command_buffer.context().create_buffer_gpu(capacity)?;
        let mut accepted_output_tokens = command_buffer.context().create_buffer_gpu(capacity)?;
        let mut accepted_output_norm = command_buffer.allocate_scratch(norm_row_bytes)?;

        self.kernel.encode(
            &trie_nodes,
            &parents,
            token_ids,
            sampled_tokens,
            output_norm,
            &mut accepted_indices,
            &mut accepted_input_tokens,
            &mut accepted_output_tokens,
            &mut accepted_output_norm,
            num_nodes,
            max_accepted + 1,
            norm_row_bytes as u32,
            command_buffer,
        );

        Ok(TrieVerification {
            accepted_indices,
            accepted_input_tokens,
            accepted_output_tokens,
            accepted_output_norm,
        })
    }
}
