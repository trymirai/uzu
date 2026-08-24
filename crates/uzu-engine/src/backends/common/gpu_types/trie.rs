use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct TrieNode {
    pub trie_start: u32,
    pub trie_end: u32,
    pub height: u32,
}
