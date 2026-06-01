use std::cmp::Ordering;

use itertools::Itertools;
use thiserror::Error;

use crate::{
    backends::common::gpu_types::trie::TrieNode as GpuTrieNode,
    data_type::DataType,
    encodable_block::sampling::{PRng, speculator_sample},
    engine::language_model::grammar::{Grammar, GrammarError},
    speculators::speculator::Speculator,
};

#[derive(Debug, Error)]
pub enum TrieError {
    #[error("child with the same token id is already present")]
    DuplicateTokenId,
}

pub struct TrieCreationConfig {
    pub width: usize,
}

impl Default for TrieCreationConfig {
    fn default() -> Self {
        TrieCreationConfig {
            width: 4,
        }
    }
}

#[derive(Debug)]
pub struct TrieNode {
    token: u64,
    seed: u64,
    next: Vec<TrieNode>,
}

#[derive(Debug)]
struct FlatTrieNode<'a> {
    node: &'a TrieNode,
    subtrie_range: (usize, usize),
    height: usize,
}

#[derive(Debug)]
pub struct FlatTrie<'a> {
    tokens: Box<[FlatTrieNode<'a>]>,
}

impl TrieNode {
    pub fn new(
        token: u64,
        seed: u64,
    ) -> Self {
        Self {
            token,
            seed,
            next: Vec::new(),
        }
    }

    pub fn add(
        &mut self,
        next: TrieNode,
    ) -> Result<usize, TrieError> {
        if self.next.iter().any(|n| n.token == next.token) {
            return Err(TrieError::DuplicateTokenId);
        }

        self.next.push(next);
        Ok(self.next.len() - 1)
    }

    pub fn get(
        &self,
        token: u64,
    ) -> Option<&TrieNode> {
        self.next.iter().find(|n| n.token == token)
    }

    pub fn get_mut(
        &mut self,
        token: u64,
    ) -> Option<&mut TrieNode> {
        self.next.iter_mut().find(|n| n.token == token)
    }

    #[cfg(test)]
    pub fn token(&self) -> u64 {
        self.token
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn from_speculator<'grammar>(
        prefix: &[u64],
        prng: &PRng,
        mut grammar: Option<&mut (dyn Grammar + 'grammar)>,
        speculator: &dyn Speculator,
        vocab_size: usize,
        creation_config: &TrieCreationConfig,
        max_length: usize,
    ) -> Self {
        assert!(max_length >= 1, "can't have zero sized trie");
        assert!(!prefix.is_empty(), "need seed node");

        let prefix_length = prefix.len();
        let mut speculated_suffix = prefix.to_vec();

        let mut length = 1;
        let mut height = 0;
        let mut root = Self::new(*prefix.last().unwrap(), prng.derive((prefix_length - 1) as u64));

        let mut cur_node = &mut root;
        let mut cur_node_width = 0;
        let mut cur_node_speculator_weights = speculator.speculate(&speculated_suffix);

        let mut next_node = None;

        while length < max_length {
            // Guuumbel speculator trick: both speculator and llm sample via gumbel max trick using the same noise for increased acceptance rate
            if let Some(next_speculated_token) =
                speculator_sample(cur_node.seed(), vocab_size, &cur_node_speculator_weights)
            {
                // Add speculated token to the trie
                if let Some(grammar) = grammar.as_deref_mut() {
                    if grammar.accept_token(next_speculated_token).is_err() {
                        cur_node_speculator_weights.remove(&next_speculated_token);
                        continue;
                    }
                    grammar.rollback(1);
                }

                let leaf_node = Self::new(next_speculated_token, prng.derive((prefix_length + height) as u64));
                cur_node.add(leaf_node).unwrap();

                // If this is the first node we sampled (most likely to be selected after gumbel noise application) - set it as the next one
                if next_node.is_none() {
                    next_node = Some(next_speculated_token);
                }

                // Remove the token so that the next iteration samples the second most likely after gumbel noise application token
                cur_node_speculator_weights.remove(&next_speculated_token);

                length += 1;
                cur_node_width += 1;
                if cur_node_width >= creation_config.width {
                    cur_node_speculator_weights.clear();
                }
            } else if let Some(next_node_token) = next_node.take() {
                // Out of speculated tokens for this node, move onto the likeliest next node
                speculated_suffix.push(next_node_token);
                if let Some(grammar) = grammar.as_deref_mut()
                    && grammar.accept_token(next_node_token).is_err()
                {
                    break;
                }
                height += 1;
                cur_node = cur_node.get_mut(next_node_token).unwrap();
                cur_node_width = 0;
                cur_node_speculator_weights = speculator.speculate(&speculated_suffix);
                continue;
            } else {
                // Dead end, exit
                break;
            };
        }

        if let Some(grammar) = grammar {
            grammar.rollback(height);
        }

        root
    }

    pub fn flat(
        prefix_length: usize,
        tokens: &[u64],
        prng: &PRng,
    ) -> Self {
        assert!(!tokens.is_empty(), "need seed node");

        let mut root = TrieNode::new(tokens[0], prng.derive(prefix_length as u64));
        let mut leaf = &mut root;

        for (index, token) in tokens.iter().copied().enumerate().skip(1) {
            leaf.add(TrieNode::new(token, prng.derive((prefix_length + index) as u64))).unwrap();
            leaf = &mut leaf.next[0];
        }

        root
    }

    pub fn linearize(&self) -> FlatTrie<'_> {
        let mut tokens = vec![FlatTrieNode::new(self, (0, 0), 0)];

        let mut stack = vec![(0, 0)];
        while let Some((cur_node_idx, next_child_idx)) = stack.last_mut() {
            let Some(next_node) = tokens[*cur_node_idx].node.next.get(*next_child_idx) else {
                tokens[*cur_node_idx].subtrie_range.1 = tokens.len() - 1;
                stack.pop();
                continue;
            };
            *next_child_idx += 1;

            tokens.push(FlatTrieNode::new(next_node, (tokens.len(), tokens.len()), stack.len()));

            if !next_node.next.is_empty() {
                stack.push((tokens.len() - 1, 0));
            }
        }

        FlatTrie::new(tokens.into_boxed_slice())
    }
}

impl<'a> FlatTrieNode<'a> {
    fn new(
        node: &'a TrieNode,
        subtrie_range: (usize, usize),
        height: usize,
    ) -> Self {
        Self {
            node,
            subtrie_range,
            height,
        }
    }
}

impl<'a> FlatTrie<'a> {
    fn new(tokens: Box<[FlatTrieNode<'a>]>) -> Self {
        Self {
            tokens,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_ids(&self) -> impl Iterator<Item = u64> {
        self.tokens.iter().map(|n| n.node.token)
    }

    pub fn token_subtrie_ranges(&self) -> impl Iterator<Item = GpuTrieNode> {
        self.tokens.iter().map(|n| {
            let (start, end) = n.subtrie_range;

            GpuTrieNode {
                trie_start: start as u32,
                trie_end: end as u32,
                height: n.height as u32,
            }
        })
    }

    pub fn token_positions(&self) -> impl Iterator<Item = usize> {
        self.tokens.iter().map(|n| n.height)
    }

    pub fn token_seeds(&self) -> impl Iterator<Item = u64> {
        self.tokens.iter().map(|n| n.node.seed)
    }

    pub fn fill_bitmasks<'grammar>(
        &self,
        bitmasks: &mut [u32],
        vocab_size: usize,
        grammar: &mut (dyn Grammar + 'grammar),
    ) -> bool {
        let vocab_size_in_u32s = vocab_size.div_ceil(DataType::U32.size_in_bits());
        assert!(bitmasks.len() == self.tokens.len() * vocab_size_in_u32s);

        let mut any_non_full = false;
        let mut last_token_height = 0;
        for ((token_index, token), bitmask) in
            self.tokens.iter().enumerate().zip_eq(bitmasks.chunks_exact_mut(vocab_size_in_u32s))
        {
            match token.height.cmp(&last_token_height) {
                Ordering::Less => {
                    grammar.rollback(last_token_height - token.height);
                },
                Ordering::Equal => {
                    assert!(token_index == 0, "non-root trie node cannot have equal height to the previous one");
                },
                Ordering::Greater => {
                    assert!(
                        last_token_height + 1 == token.height,
                        "trie node cannot be higher than prev trie node + 1"
                    );
                    grammar.accept_token(token.node.token).expect("flat trie doesn't match grammar");
                },
            }
            last_token_height = token.height;

            any_non_full |= grammar.next_bitmask(bitmask);
        }

        if last_token_height > 0 {
            grammar.rollback(last_token_height);
        }

        any_non_full
    }

    pub fn root(&self) -> Option<&TrieNode> {
        self.tokens.first().map(|n| n.node)
    }

    pub fn index(
        &self,
        node: &'a TrieNode,
    ) -> Option<usize> {
        self.tokens.iter().position(|n| std::ptr::eq(n.node, node))
    }

    pub fn accept<'grammar>(
        &self,
        sampled_tokens: &[u64],
        mut grammar: Option<&mut (dyn Grammar + 'grammar)>,
    ) -> Result<Box<[(usize, u64)]>, GrammarError> {
        let mut current_token = self.root().unwrap();
        let mut accepted = Vec::new();
        loop {
            let current_token_index = self.index(current_token).unwrap();
            let current_token_id = sampled_tokens[current_token_index];

            accepted.push((current_token_index, current_token_id));
            if let Some(grammar) = grammar.as_deref_mut()
                && !grammar.is_terminated()
            {
                grammar.accept_token(current_token_id)?;
            }

            let Some(next_token) = current_token.get(current_token_id) else {
                break;
            };

            if let Some(grammar) = grammar.as_deref_mut() {
                assert!(!grammar.is_terminated(), "Grammar has terminated but llm continued generation");
            }

            current_token = next_token;
        }

        Ok(accepted.into_boxed_slice())
    }
}

#[cfg(test)]
#[path = "../unit/trie_test.rs"]
mod tests;
