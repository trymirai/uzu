use std::collections::HashMap;

use super::{node::Node, speculated_suffix::SpeculatedSuffix};
use crate::{llm::rng::DerivableSeed, speculators::speculator::Speculator};

pub struct TrieCreationConfig {
    pub top_k: usize,
    pub prob_cutoff: f32,
}

impl Default for TrieCreationConfig {
    fn default() -> Self {
        TrieCreationConfig {
            top_k: 2,
            prob_cutoff: 0.1,
        }
    }
}

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

    pub fn from_speculator(
        prefix: &[u64],
        seed: &mut DerivableSeed,
        include_last_prefix_token: bool,
        speculator: &dyn Speculator,
        creation_config: &TrieCreationConfig,
        max_length: usize,
    ) -> Self {
        let mut trie = Self::new();
        let mut current_node = &mut trie.root;
        let mut trie_size = 0;

        if include_last_prefix_token {
            current_node = current_node
                .get_or_insert_next(*prefix.last().unwrap(), seed.next());
            trie_size += 1;
        }

        let mut speculation_prefix = prefix.to_vec();
        while trie_size < max_length {
            let speculation_result = speculator.speculate(&speculation_prefix);

            let mut arr =
                speculation_result.into_iter().collect::<Vec<(u64, f32)>>();
            arr.sort_by(|(_, a), (_, b)| f32::total_cmp(b, a));

            let arr = arr
                .into_iter()
                .enumerate()
                .filter(|&(i, (_, p))| {
                    i == 0 || p > creation_config.prob_cutoff
                })
                .map(|(_, x)| x)
                .take(usize::min(creation_config.top_k, max_length - trie_size))
                .collect::<Vec<(u64, f32)>>();

            let Some(&(top_token, _)) = arr.first() else {
                break;
            };

            trie_size += arr.len();
            for &(tok, _) in arr.iter() {
                current_node.get_or_insert_next(tok, seed.next());
            }

            current_node = current_node.get_or_insert_next(top_token, 0);
            speculation_prefix.push(top_token);
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
            current_node = current_node.get_or_insert_next(token, 0);
        }
    }

    pub fn linearize(
        &self,
        start_index: usize,
        max_length: usize,
    ) -> SpeculatedSuffix {
        let mut tokens = Vec::new();
        let mut seeds = Vec::new();
        let mut indices = Vec::new();

        self.root.dfs(|path| {
            if tokens.len() >= max_length {
                return;
            }
            if !path.is_empty() {
                tokens.push(path.last().unwrap().1);
                seeds.push(path.last().unwrap().2);
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

            let (current_index, current_token, _) = *path.last().unwrap();
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
            seeds,
            transition_map,
        }
    }
}
