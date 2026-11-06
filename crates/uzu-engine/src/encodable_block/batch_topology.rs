use crate::{
    backends::common::{Backend, CommandBuffer, CommandBufferEncoding, Context},
    data_type::DataType,
};

pub struct BatchTopology<B: Backend> {
    /// Present iff the batch is a trie.
    nodes: Option<B::GlobalGpuBuffer>,
    /// Present unless on prefill.
    token_ids: Option<B::GlobalGpuBuffer>,
    /// Present iff (trie && stochastic sampling).
    seeds: Option<B::GlobalGpuBuffer>,
    max_size: u32,
    full_accept: bool,
    is_flat: bool,
}

impl<B: Backend> BatchTopology<B> {
    pub fn flat(size: u32) -> Self {
        Self::new(None, None, None, size, true, true)
    }

    pub(crate) fn with_token_ids(
        mut self,
        command_buffer: &mut <<B as Backend>::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self, <B as Backend>::Error> {
        let context = command_buffer.context();
        let word_bytes = DataType::U32.size_in_bytes();
        self.token_ids = Some(context.create_buffer_gpu(self.max_size as usize * word_bytes)?);
        Ok(self)
    }

    pub(crate) fn new(
        nodes: Option<B::GlobalGpuBuffer>,
        token_ids: Option<B::GlobalGpuBuffer>,
        seeds: Option<B::GlobalGpuBuffer>,
        max_size: u32,
        full_accept: bool,
        is_flat: bool,
    ) -> Self {
        let is_trie = nodes.is_some();
        assert_eq!(full_accept, !is_trie, "a batch is either a trie or fully accepted");
        assert!(is_trie || is_flat, "node-less batches are sequential");
        assert!(!is_trie || token_ids.is_some(), "trie batches carry token ids");
        Self {
            nodes,
            token_ids,
            seeds,
            max_size,
            full_accept,
            is_flat,
        }
    }

    pub fn size(&self) -> u32 {
        self.max_size
    }

    pub fn full_accept(&self) -> bool {
        self.full_accept
    }

    pub fn is_flat(&self) -> bool {
        self.is_flat
    }

    pub fn nodes(&self) -> Option<&B::GlobalGpuBuffer> {
        self.nodes.as_ref()
    }

    pub fn token_ids(&self) -> Option<&B::GlobalGpuBuffer> {
        self.token_ids.as_ref()
    }

    pub(crate) fn token_ids_mut(&mut self) -> &mut Option<B::GlobalGpuBuffer> {
        &mut self.token_ids
    }

    pub fn seeds(&self) -> Option<&B::GlobalGpuBuffer> {
        self.seeds.as_ref()
    }
}
