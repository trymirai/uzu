use std::collections::HashMap;

use super::{node::Node, speculated_suffix::SpeculatedSuffix};

#[derive(Debug)]
pub struct TokenTrie {
    root: Node,
}

impl Default for TokenTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenTrie {
    pub fn new() -> Self {
        Self {
            root: Node::new(),
        }
    }

    pub fn from_sequences(sequences: &[Vec<u64>]) -> Self {
        let mut trie = Self::new();
        for sequence in sequences {
            trie.insert(sequence);
        }
        trie
    }

    pub fn insert<'a, I>(
        &mut self,
        sequence: I,
    ) where
        I: IntoIterator<Item = &'a u64>,
    {
        let mut current_node = &mut self.root;
        for &token in sequence {
            current_node = current_node.get_or_insert_next(token);
        }
    }

    pub fn linearize(
        &self,
        start_index: usize,
        max_length: usize,
    ) -> SpeculatedSuffix {
        let mut tokens = Vec::new();
        let mut indices = Vec::new();

        self.root.dfs(|path| {
            if tokens.len() >= max_length {
                return;
            }
            if !path.is_empty() {
                tokens.push(path.last().unwrap().1);
                indices.push(start_index + path.len() - 1);
            }
        });

        let tokens_len = tokens.len();

        let mut transition_map: HashMap<
            isize,
            std::collections::HashMap<u64, isize>,
        > = HashMap::new();
        for index in -1..(tokens_len as isize) {
            transition_map.insert(index, HashMap::new());
        }

        self.root.dfs(|path| {
            if path.is_empty()
                || path.last().unwrap().0 >= (max_length as isize)
            {
                return;
            }

            let (current_index, current_token) = *path.last().unwrap();
            let parent_index = if path.len() > 1 {
                path[path.len() - 2].0
            } else {
                -1
            };

            if let Some(transitions) = transition_map.get_mut(&parent_index) {
                transitions.insert(current_token, current_index);
            }
        });

        SpeculatedSuffix {
            tokens,
            indices,
            transition_map,
        }
    }
}
