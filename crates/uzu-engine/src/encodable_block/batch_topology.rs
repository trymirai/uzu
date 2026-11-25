use std::ops::Range;

use crate::backends::common::Backend;

pub enum BatchKind<B: Backend> {
    Sequence,
    Trie {
        nodes: B::GlobalGpuBuffer,
        is_flat: bool,
    },
    Image {
        width: u32,
        height: u32,
        merge_size: u32,
        first_cell: u32,
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

    pub fn image(
        width: u32,
        height: u32,
        merge_size: u32,
        cells: Range<u32>,
    ) -> Self {
        Self {
            kind: BatchKind::Image {
                width,
                height,
                merge_size,
                first_cell: cells.start,
            },
            size: cells.end - cells.start,
        }
    }

    pub fn kind(&self) -> &BatchKind<B> {
        &self.kind
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn full_accept(&self) -> bool {
        !matches!(self.kind, BatchKind::Trie { .. })
    }

    pub fn is_flat(&self) -> bool {
        match self.kind {
            BatchKind::Trie {
                is_flat,
                ..
            } => is_flat,
            BatchKind::Sequence
            | BatchKind::Image {
                ..
            } => true,
        }
    }

    pub fn nodes(&self) -> Option<&B::GlobalGpuBuffer> {
        match &self.kind {
            BatchKind::Trie {
                nodes,
                ..
            } => Some(nodes),
            BatchKind::Sequence
            | BatchKind::Image {
                ..
            } => None,
        }
    }
}
