#[cfg(grammar)]
use itertools::Itertools;
use thiserror::Error;

use crate::{backends::common::gpu_types::trie::TrieNode as GpuTrieNode, encodable_block::sampling::PRng};
#[cfg(grammar)]
use crate::{
    data_type::DataType,
    engine::language_model::grammar::{Grammar, GrammarError},
};

#[derive(Debug, Error)]
pub enum TrieError {
    #[error("child with the same token id is already present")]
    DuplicateTokenId,
}

#[derive(Debug, Error)]
pub enum TrieAcceptError {
    #[cfg(grammar)]
    #[error("Grammar error: {0}")]
    Grammar(#[from] GrammarError),
}

#[derive(Debug)]
pub struct TrieNode {
    token: u64,
    seed: u64,
    logprob: f32,
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
        logprob: f32,
    ) -> Self {
        Self {
            token,
            seed,
            logprob,
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

    #[cfg(test)]
    pub fn token(&self) -> u64 {
        self.token
    }

    #[cfg(test)]
    pub fn logprob(&self) -> f32 {
        self.logprob
    }

    #[cfg(test)]
    pub fn node_count(&self) -> usize {
        1 + self.next.iter().map(TrieNode::node_count).sum::<usize>()
    }

    pub fn prune_to_budget(
        &mut self,
        budget: usize,
    ) {
        assert!(budget > 0, "budget must keep at least the root");

        fn collect_logprobs(
            node: &TrieNode,
            parent_logprob: f32,
            logprobs: &mut Vec<f32>,
        ) {
            let logprob = parent_logprob + node.logprob;
            logprobs.push(logprob);
            for child in &node.next {
                collect_logprobs(child, logprob, logprobs);
            }
        }
        let mut logprobs = Vec::new();
        collect_logprobs(self, 0.0, &mut logprobs);
        if budget >= logprobs.len() {
            return;
        }

        let mut order: Box<[usize]> = (0..logprobs.len()).collect();
        order.sort_by(|&a, &b| logprobs[b].total_cmp(&logprobs[a]));
        let mut kept = vec![false; logprobs.len()];
        for &index in order.iter().take(budget) {
            kept[index] = true;
        }

        fn prune(
            node: &mut TrieNode,
            kept: &[bool],
            cursor: &mut usize,
        ) {
            *cursor += 1;
            let mut children = std::mem::take(&mut node.next);
            children.retain_mut(|child| {
                let child_index = *cursor;
                prune(child, kept, cursor);
                kept[child_index]
            });
            node.next = children;
        }
        prune(self, &kept, &mut 0);
    }

    #[cfg(test)]
    pub fn prune_by_logprob_threshold(
        &mut self,
        min_logprob: f32,
    ) {
        fn prune(
            node: &mut TrieNode,
            parent_logprob: f32,
            min_logprob: f32,
        ) {
            node.next.retain_mut(|child| {
                let logprob = parent_logprob + child.logprob;
                let keep = logprob >= min_logprob;
                if keep {
                    prune(child, logprob, min_logprob);
                }
                keep
            });
        }
        prune(self, 0.0, min_logprob);
    }

    #[cfg(test)]
    pub fn prune_to_depth(
        &mut self,
        max_depth: usize,
    ) {
        if max_depth == 0 {
            self.next.clear();
            return;
        }
        for child in &mut self.next {
            child.prune_to_depth(max_depth - 1);
        }
    }

    pub fn flat(
        prefix_length: usize,
        tokens: &[u64],
        prng: &PRng,
    ) -> Self {
        assert!(!tokens.is_empty(), "need seed node");

        let mut root = TrieNode::new(tokens[0], prng.derive(prefix_length as u64), 0.0);
        let mut leaf = &mut root;

        for (index, token) in tokens.iter().copied().enumerate().skip(1) {
            leaf.add(TrieNode::new(token, prng.derive((prefix_length + index) as u64), 0.0)).unwrap();
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

    pub fn token_seeds(&self) -> impl Iterator<Item = u64> {
        self.tokens.iter().map(|n| n.node.seed)
    }

    #[cfg(grammar)]
    pub fn fill_bitmasks(
        &self,
        bitmasks: &mut [u32],
        vocab_size: usize,
        grammar: &mut Grammar,
    ) -> bool {
        let vocab_size_in_u32s = vocab_size.div_ceil(DataType::U32.size_in_bits());
        assert!(bitmasks.len() == self.tokens.len() * vocab_size_in_u32s);

        let mut any_non_full = false;
        let mut last_token_height = 0;
        for ((token_index, token), bitmask) in
            self.tokens.iter().enumerate().zip_eq(bitmasks.chunks_exact_mut(vocab_size_in_u32s))
        {
            if token_index > 0 {
                if token.height <= last_token_height {
                    grammar.rollback(last_token_height - token.height + 1);
                }
                grammar.accept_token(token.node.token).expect("flat trie doesn't match grammar");
            }

            any_non_full |= grammar.next_bitmask(bitmask);

            last_token_height = token.height;
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

    pub fn accept(
        &self,
        sampled_tokens: &[u64],
        #[cfg(grammar)] mut grammar: Option<&mut Grammar>,
    ) -> Result<Box<[(usize, u64, u64)]>, TrieAcceptError> {
        let mut current_token = self.root().unwrap();
        let mut accepted = Vec::new();
        loop {
            let current_token_index = self.index(current_token).unwrap();
            let current_token_id = sampled_tokens[current_token_index];

            accepted.push((current_token_index, current_token.token, current_token_id));
            #[cfg(grammar)]
            if let Some(grammar) = grammar.as_mut()
                && !grammar.is_terminated()
            {
                grammar.accept_token(current_token_id)?;
            }

            let Some(next_token) = current_token.get(current_token_id) else {
                break;
            };

            #[cfg(grammar)]
            if let Some(grammar) = grammar.as_mut() {
                assert!(!grammar.is_terminated(), "Grammar has terminated but llm continued generation");
            }

            current_token = next_token;
        }

        Ok(accepted.into_boxed_slice())
    }
}

#[cfg(test)]
#[path = "../tests/unit/trie_test.rs"]
mod tests;
