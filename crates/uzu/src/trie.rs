use crate::{
    generator::{
        grammar::CompiledGrammar, gumbel::speculator_sample, rng::DerivableSeed,
    },
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
    mask: Option<Vec<u32>>,
    seed: u64,
    next: Vec<TrieNode>,
}

#[derive(Debug)]
pub struct FlatTrie<'a> {
    tokens: Vec<(&'a TrieNode, usize)>,
    has_masks: bool,
}

impl TrieNode {
    pub fn new(
        token: u64,
        mask: Option<Vec<u32>>,
        seed: u64,
    ) -> Self {
        Self {
            token,
            mask,
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
        mut compiled_grammar: Option<&mut CompiledGrammar>,
        speculator: &dyn Speculator,
        creation_config: &TrieCreationConfig,
        max_length: usize,
    ) -> Self {
        assert!(max_length >= 1, "can't have zero sized trie");
        assert!(prefix.len() >= 1, "need seed node");

        let mut speculated_suffix = prefix.to_vec();

        let mut length = 1;
        let mut grammar_accept_count = 0;
        let mask =
            if let Some(compiled_grammar) = compiled_grammar.as_deref_mut() {
                Some(compiled_grammar.next_bitmask().unwrap())
            } else {
                None
            };

        let mut root =
            Self::new(*prefix.last().unwrap(), mask, next_seed.next());

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
                let mask = if let Some(compiled_grammar) =
                    compiled_grammar.as_deref_mut()
                {
                    compiled_grammar.accept_token(next_speculated_token);
                    let next_bitmask = compiled_grammar.next_bitmask().unwrap();
                    compiled_grammar.rollback(1);

                    Some(next_bitmask)
                } else {
                    None
                };

                let leaf_node =
                    Self::new(next_speculated_token, mask, next_seed.next());
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
                if let Some(compiled_grammar) = compiled_grammar.as_deref_mut()
                {
                    compiled_grammar.accept_token(next_node_token);
                    grammar_accept_count += 1;
                }
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

        if let Some(compiled_grammar) = compiled_grammar.as_deref_mut() {
            compiled_grammar.rollback(grammar_accept_count);
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

        FlatTrie::new(tokens)
    }
}

impl<'a> FlatTrie<'a> {
    pub fn new(tokens: Vec<(&'a TrieNode, usize)>) -> Self {
        let all_masks = tokens.iter().all(|(n, _p)| n.mask.is_some());
        let none_masks = tokens.iter().all(|(n, _p)| n.mask.is_none());
        assert!(
            all_masks || none_masks,
            "flat trie nodes should either all have masks or none can have masks"
        );

        Self {
            tokens,
            has_masks: all_masks,
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

    pub fn token_masks(&self) -> Option<impl Iterator<Item = &[u32]>> {
        if self.has_masks {
            Some(self.tokens.iter().map(|&(n, _p)| n.mask.as_deref().unwrap()))
        } else {
            None
        }
    }

    pub fn token_seeds(&self) -> impl Iterator<Item = u64> {
        self.tokens.iter().map(|&(n, _p)| n.seed)
    }

    pub fn index(
        &self,
        node: &'a TrieNode,
    ) -> Option<usize> {
        self.tokens.iter().position(|&(n, _)| std::ptr::eq(n, node))
    }
}
