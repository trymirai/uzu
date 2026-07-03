use crate::backends::common::gpu_types::trie::TrieNode;

pub struct BatchTopology<'a> {
    nodes: &'a [TrieNode],
    parents: Box<[i32]>,
    is_flat: bool,
    full_accept: bool,
}

impl<'a> BatchTopology<'a> {
    pub fn new(
        nodes: &'a [TrieNode],
        full_accept: bool,
    ) -> Self {
        let mut stack: Vec<usize> = Vec::new();

        let mut parents = Box::new_uninit_slice(nodes.len());
        let mut is_flat = true;

        for (index, node) in nodes.iter().enumerate() {
            stack.truncate(node.height as usize);
            parents[index].write(stack.last().map(|i| *i as i32).unwrap_or(-1));
            stack.push(index);

            is_flat &= node.height == index as u32;
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

    pub fn nodes(&self) -> &'a [TrieNode] {
        self.nodes
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_flat(&self) -> bool {
        self.is_flat
    }

    pub fn full_accept(&self) -> bool {
        self.full_accept
    }

    pub fn heights(&self) -> impl Iterator<Item = u32> + 'a {
        self.nodes.iter().map(|trie| trie.height)
    }

    pub fn parents(&self) -> &[i32] {
        &self.parents
    }
}
