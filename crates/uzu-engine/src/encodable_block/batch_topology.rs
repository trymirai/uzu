use crate::backends::common::Backend;

pub enum BatchKind<B: Backend> {
    /// Rows continue each other.
    Sequence,
    /// Rows are nodes of a trie of continuations; `is_flat` when the trie is a single chain.
    Trie {
        nodes: B::GlobalGpuBuffer,
        is_flat: bool,
    },
}

pub struct BatchTopology<B: Backend> {
    kind: BatchKind<B>,
    size: u32,
}

impl<B: Backend> BatchTopology<B> {
    pub fn flat(size: u32) -> Self {
        Self {
            kind: BatchKind::Sequence,
            size,
        }
    }

    pub fn trie(
        nodes: B::GlobalGpuBuffer,
        size: u32,
        is_flat: bool,
    ) -> Self {
        Self {
            kind: BatchKind::Trie {
                nodes,
                is_flat,
            },
            size,
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn full_accept(&self) -> bool {
        !matches!(self.kind, BatchKind::Trie { .. })
    }

    pub fn is_flat(&self) -> bool {
        match self.kind {
            BatchKind::Sequence => true,
            BatchKind::Trie {
                is_flat,
                ..
            } => is_flat,
        }
    }

    pub fn nodes(&self) -> Option<&B::GlobalGpuBuffer> {
        match &self.kind {
            BatchKind::Trie {
                nodes,
                ..
            } => Some(nodes),
            BatchKind::Sequence => None,
        }
    }
}
