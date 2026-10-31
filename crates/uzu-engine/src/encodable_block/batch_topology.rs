use crate::backends::common::Backend;

pub enum BatchTopology<B: Backend> {
    Flat {
        size: u32,
    },
    Tree {
        nodes: B::GlobalGpuBuffer,
        parents: B::GlobalGpuBuffer,
        heights: B::GlobalGpuBuffer,
        size: u32,
    },
}

impl<B: Backend> BatchTopology<B> {
    pub fn flat(size: u32) -> Self {
        Self::Flat {
            size,
        }
    }

    pub fn size(&self) -> u32 {
        match self {
            Self::Flat {
                size,
            } => *size,
            Self::Tree {
                size,
                ..
            } => *size,
        }
    }

    pub fn is_flat(&self) -> bool {
        matches!(self, Self::Flat { .. })
    }

    pub fn full_accept(&self) -> bool {
        self.is_flat()
    }

    pub fn nodes(&self) -> Option<&B::GlobalGpuBuffer> {
        match self {
            Self::Flat {
                ..
            } => None,
            Self::Tree {
                nodes,
                ..
            } => Some(nodes),
        }
    }

    pub fn parents(&self) -> Option<&B::GlobalGpuBuffer> {
        match self {
            Self::Flat {
                ..
            } => None,
            Self::Tree {
                parents,
                ..
            } => Some(parents),
        }
    }

    pub fn heights(&self) -> Option<&B::GlobalGpuBuffer> {
        match self {
            Self::Flat {
                ..
            } => None,
            Self::Tree {
                heights,
                ..
            } => Some(heights),
        }
    }
}
