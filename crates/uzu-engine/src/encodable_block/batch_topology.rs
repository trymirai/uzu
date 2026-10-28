use crate::backends::common::gpu_types::trie::TrieNode;

pub struct BatchTopology {
    nodes: Box<[TrieNode]>,
    parents: Box<[i32]>,
    is_flat: bool,
    full_accept: bool,
}

impl BatchTopology {
    pub fn new(
        nodes: Box<[TrieNode]>,
        full_accept: bool,
    ) -> Self {
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
        assert!(is_flat || !full_accept, "full_accept requires trie to be flat");

        Self {
            nodes,
            parents,
            is_flat,
            full_accept,
        }
    }

    pub fn flat(
        size: u32,
        full_accept: bool,
    ) -> Self {
        let nodes = (0..size)
            .map(|index| TrieNode {
                trie_start: index,
                trie_end: size - 1,
                height: index,
            })
            .collect::<Box<[_]>>();

        Self::new(nodes, full_accept)
    }

    pub fn size(&self) -> u32 {
        self.nodes.len() as u32
    }

    pub fn is_flat(&self) -> bool {
        self.is_flat
    }

    pub fn full_accept(&self) -> bool {
        self.full_accept
    }

    pub fn nodes(&self) -> &[TrieNode] {
        &self.nodes
    }

    pub fn parents(&self) -> &[i32] {
        &self.parents
    }

    pub fn heights(&self) -> impl Iterator<Item = u32> {
        self.nodes().iter().map(|t| t.height)
    }
}
