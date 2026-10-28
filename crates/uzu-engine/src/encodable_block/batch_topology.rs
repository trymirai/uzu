use crate::backends::common::gpu_types::trie::TrieNode;

pub enum BatchTopology {
    Flat {
        size: u32,
    },
    Tree {
        nodes: Box<[TrieNode]>,
        parents: Box<[i32]>,
        is_flat: bool,
    },
}

impl BatchTopology {
    pub fn new(nodes: Box<[TrieNode]>) -> Self {
        let mut stack: Vec<usize> = Vec::new();

        let mut parents = Box::new_uninit_slice(nodes.len());
        let mut is_flat = true;

        for (node, index) in nodes.iter().zip(0u32..) {
            stack.truncate(node.height as usize);
            parents[index as usize].write(stack.last().map(|i| *i as i32).unwrap_or(-1));
            stack.push(index as usize);

            is_flat &= node.height == index;
        }
        let parents = unsafe { parents.assume_init() };

        Self::Tree {
            nodes,
            parents,
            is_flat,
        }
    }

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
                nodes,
                ..
            } => nodes.len() as u32,
        }
    }

    pub fn is_flat(&self) -> bool {
        match self {
            Self::Flat {
                ..
            } => true,
            Self::Tree {
                is_flat,
                ..
            } => *is_flat,
        }
    }

    pub fn full_accept(&self) -> bool {
        matches!(self, Self::Flat { .. })
    }

    pub fn nodes(&self) -> Option<&[TrieNode]> {
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

    pub fn parents(&self) -> Option<&[i32]> {
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

    pub fn heights(&self) -> impl Iterator<Item = u32> {
        let mut inner = self.nodes().map(|nodes| nodes.iter().map(|t| t.height));
        let mut fallback = 0..self.size();
        std::iter::from_fn(move || inner.as_mut().map_or_else(|| fallback.next(), |it| it.next()))
    }
}
