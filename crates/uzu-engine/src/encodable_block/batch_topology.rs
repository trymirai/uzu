use std::mem::size_of;

use crate::{
    backends::common::{Backend, CommandBuffer, CommandBufferEncoding, Context, gpu_types::trie::TrieNode},
    data_type::DataType,
};

pub struct BatchTopology<B: Backend> {
    pub(crate) nodes: B::GlobalGpuBuffer,
    pub(crate) token_ids: B::GlobalGpuBuffer,
    pub(crate) seeds: B::GlobalGpuBuffer,
    pub(crate) count: B::GlobalGpuBuffer,
    pub(crate) max_size: u32,
    pub(crate) is_flat: bool,
    pub(crate) full_accept: bool,
}

impl<B: Backend> BatchTopology<B> {
    pub fn new(
        nodes: Box<[TrieNode]>,
        full_accept: bool,
    ) -> Self {
        todo!()
    }

    pub fn flat(
        size: u32,
        full_accept: bool,
        command_buffer: &mut <<B as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self, <B as Backend>::Error> {
        let context = command_buffer.context();
        let word_bytes = DataType::U32.size_in_bytes();
        Ok(Self {
            nodes: context.create_buffer_gpu(size as usize * size_of::<TrieNode>())?,
            token_ids: context.create_buffer_gpu(size as usize * word_bytes)?,
            seeds: context.create_buffer_gpu(size as usize * size_of::<u64>())?,
            count: context.create_buffer_gpu(word_bytes)?,
            max_size: size,
            is_flat: true,
            full_accept,
        })
    }

    pub fn from_buffers(
        nodes: B::GlobalGpuBuffer,
        token_ids: B::GlobalGpuBuffer,
        seeds: B::GlobalGpuBuffer,
        count: B::GlobalGpuBuffer,
        max_size: u32,
        is_flat: bool,
        full_accept: bool,
    ) -> Self {
        Self {
            nodes,
            token_ids,
            seeds,
            count,
            max_size,
            is_flat,
            full_accept,
        }
    }

    pub fn size(&self) -> u32 {
        self.max_size
    }

    pub fn is_flat(&self) -> bool {
        self.is_flat
    }

    pub fn full_accept(&self) -> bool {
        self.full_accept
    }

    pub fn nodes(&self) -> &B::GlobalGpuBuffer {
        &self.nodes
    }

    pub fn token_ids(&self) -> &B::GlobalGpuBuffer {
        &self.token_ids
    }

    pub fn seeds(&self) -> &B::GlobalGpuBuffer {
        &self.seeds
    }

    pub fn count(&self) -> &B::GlobalGpuBuffer {
        &self.count
    }
}
