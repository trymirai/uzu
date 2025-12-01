use crate::{
    generator::{gumbel::speculator_sample, rng::DerivableSeed},
    speculators::speculator::Speculator,
};

use thiserror::Error;

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
pub struct FlatTrie<'a> {
    tokens: Box<[(&'a TrieNode, usize)]>,
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

    pub fn token(&self) -> u64 {
        self.token
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn from_speculator(
        prefix: &[u64],
        next_seed: &mut DerivableSeed,
        speculator: &dyn Speculator,
        creation_config: &TrieCreationConfig,
        max_length: usize,
    ) -> Self {
        assert!(max_length >= 1, "can't have zero sized trie");
        assert!(prefix.len() >= 1, "need seed node");

        let mut speculated_suffix = prefix.to_vec();

        let mut length = 1;
        let mut root = Self::new(*prefix.last().unwrap(), next_seed.next());

        let mut cur_node = &mut root;
        let mut cur_node_width = 0;
        let mut cur_node_speculator_weights =
            speculator.speculate(&speculated_suffix);

        let mut next_node = None;

        while length < max_length {
            // Guuumbel speculator trick: both speculator and llm sample via gumbel max trick using the same noise for increased acceptance rate
            if let Some(next_speculated_token) =
                speculator_sample(cur_node.seed(), &cur_node_speculator_weights)
            {
                // Add speculated token to the trie
                let leaf_node =
                    Self::new(next_speculated_token, next_seed.next());
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
                cur_node = cur_node.get_mut(next_node_token).unwrap();
                cur_node_width = 0;
                cur_node_speculator_weights =
                    speculator.speculate(&speculated_suffix);
                continue;
            } else {
                // Dead end, exit
                break;
            };
        }

        root
    }

    pub fn linearize(&self) -> FlatTrie<'_> {
        let mut tokens = vec![(self, 0)];

        let mut stack = vec![(self, 0)];
        while let Some((cur_node, next_child_idx)) = stack.last_mut() {
            let Some(next_node) = cur_node.next.get(*next_child_idx) else {
                stack.pop();
                continue;
            };
            *next_child_idx += 1;

            tokens.push((next_node, stack.len()));

            if !next_node.next.is_empty() {
                stack.push((next_node, 0));
            }
        }

        FlatTrie::new(tokens.into_boxed_slice())
    }
}

impl<'a> FlatTrie<'a> {
    pub fn new(tokens: Box<[(&'a TrieNode, usize)]>) -> Self {
        Self {
            tokens,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_ids(&self) -> impl Iterator<Item = u64> {
        self.tokens.iter().map(|&(n, _p)| n.token)
    }

    pub fn token_positions(&self) -> impl Iterator<Item = usize> {
        self.tokens.iter().map(|&(_n, p)| p)
    }

    pub fn token_seeds(&self) -> impl Iterator<Item = u64> {
        self.tokens.iter().map(|&(n, _p)| n.seed)
    }

    pub fn root(&self) -> Option<&TrieNode> {
        self.tokens.first().map(|&(n, _p)| n)
    }

    pub fn index(
        &self,
        node: &'a TrieNode,
    ) -> Option<usize> {
        self.tokens.iter().position(|&(n, _)| std::ptr::eq(n, node))
    }

    pub fn accept(
        &self,
        sampled_tokens: &[u64],
    ) -> (Vec<u64>, Vec<usize>) {
        let mut current_token = self.root().unwrap();
        let mut accepted_tokens = Vec::new();
        let mut accepted_token_indices = Vec::new();
        loop {
            let current_token_index = self.index(current_token).unwrap();
            let current_token_id = sampled_tokens[current_token_index];

            accepted_token_indices.push(current_token_index);
            accepted_tokens.push(current_token_id);

            let Some(next_token) = current_token.get(current_token_id) else {
                break;
            };

            current_token = next_token;
        }

        (accepted_tokens, accepted_token_indices)
    }
}
