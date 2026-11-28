use std::ops::Range;

use super::cell_grid::CellGrid;
use crate::backends::common::Backend;

pub enum BatchKind<B: Backend> {
    Flat,
    Image {
        grid: CellGrid,
        first_cell: u32,
    },
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
            kind: BatchKind::Flat,
            size,
        }
    }

    pub fn image(
        grid: CellGrid,
        cells: Range<u32>,
    ) -> Self {
        Self {
            kind: BatchKind::Image {
                grid,
                first_cell: cells.start,
            },
            size: cells.end - cells.start,
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

    pub fn kind(&self) -> &BatchKind<B> {
        &self.kind
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    /// How far the RoPE position moves when `accepted` rows of this batch are accepted
    pub fn position_advance(
        &self,
        accepted: u32,
    ) -> u32 {
        match self.kind {
            BatchKind::Image {
                grid,
                first_cell,
            } if first_cell + accepted == grid.cell_count() => grid.width.max(grid.height),
            BatchKind::Image {
                ..
            } => 0,
            BatchKind::Flat
            | BatchKind::Trie {
                ..
            } => accepted,
        }
    }

    pub fn full_accept(&self) -> bool {
        !matches!(self.kind, BatchKind::Trie { .. })
    }

    pub fn is_flat(&self) -> bool {
        match self.kind {
            BatchKind::Flat
            | BatchKind::Image {
                ..
            } => true,
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
            BatchKind::Flat
            | BatchKind::Image {
                ..
            } => None,
        }
    }
}
