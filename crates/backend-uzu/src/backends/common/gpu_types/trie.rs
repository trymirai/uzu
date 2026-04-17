#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TrieNode {
    pub trie_start: u32,
    pub trie_end: u32,
    pub height: u32,
}
