use crate::backends::common::Backend;

pub struct BatchTopology<B: Backend> {
    /// Present iff the batch is a trie.
    nodes: Option<B::GlobalGpuBuffer>,
    size: u32,
    is_flat: bool,
    full_accept: bool,
}

impl<B: Backend> BatchTopology<B> {
    pub fn flat(size: u32) -> Self {
        Self::new(None, size, true, true)
    }

    pub(crate) fn new(
        nodes: Option<B::GlobalGpuBuffer>,
        size: u32,
        full_accept: bool,
        is_flat: bool,
    ) -> Self {
        let is_trie = nodes.is_some();
        assert_eq!(full_accept, !is_trie, "a batch is either a trie or fully accepted");
        assert!(is_trie || is_flat, "node-less batches are sequential");
        Self {
            nodes,
            size,
            full_accept,
            is_flat,
        }
    }

    pub fn size(&self) -> u32 {
        self.size
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
}
