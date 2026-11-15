use crate::backends::common::gpu_types::trie::TrieNode;

pub struct BatchTopology {
    nodes: Box<[TrieNode]>,
    parents: Box<[i32]>,
    is_flat: bool,
    full_accept: bool,
    is_set_attention: bool,
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
            is_set_attention: false,
        }
    }

    /// Build a Set Attention topology for a packed row `[state, group_1, group_2, ...]`.
    /// - `groups`: length `seq_len`, `-1` on state tokens (`0..state_len`), `1..=N` on question tokens.
    /// - `nodes`: `height` stores `state_len`; `trie_start..=trie_end` stores `[0, seq_len-1]` for state tokens
    ///   and `[group_start, group_end]` for question tokens.
    /// - `parents`: linear causal chain `[-1, 0, 1, ..., seq_len-2]` for causal ShortConv layers.
    pub fn set_attention(
        state_len: u32,
        groups: &[i32],
    ) -> Self {
        let size = groups.len();
        assert!(size > 0, "set_attention requires non-empty groups");
        let last = (size - 1) as u32;
        let mut group_bounds = std::collections::HashMap::<i32, (u32, u32)>::new();
        for (idx, &g) in groups.iter().enumerate() {
            if g >= 0 {
                let pos = idx as u32;
                group_bounds
                    .entry(g)
                    .and_modify(|(s, e)| {
                        *s = (*s).min(pos);
                        *e = (*e).max(pos);
                    })
                    .or_insert((pos, pos));
            }
        }
        let nodes: Box<[TrieNode]> = groups
            .iter()
            .enumerate()
            .map(|(idx, &g)| {
                if (idx as u32) < state_len || g < 0 {
                    TrieNode {
                        trie_start: 0,
                        trie_end: last,
                        height: state_len,
                    }
                } else {
                    let (s, e) = group_bounds[&g];
                    TrieNode {
                        trie_start: s,
                        trie_end: e,
                        height: state_len,
                    }
                }
            })
            .collect();
        let parents: Box<[i32]> = (0..size as i32).map(|i| i - 1).collect();
        Self {
            nodes,
            parents,
            is_flat: false,
            full_accept: false,
            is_set_attention: true,
        }
    }

    pub fn is_set_attention(&self) -> bool {
        self.is_set_attention
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
            .collect();

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

    pub fn heights(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.nodes.len()).map(|i| {
            if self.is_set_attention {
                i as u32
            } else {
                self.nodes[i].height
            }
        })
    }
}
