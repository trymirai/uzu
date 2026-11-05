use itertools::Itertools;
use thiserror::Error;

use crate::{
    backends::common::gpu_types::trie::TrieNode as GpuTrieNode,
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

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_ids(&self) -> impl Iterator<Item = u64> {
        self.tokens.iter().map(|n| n.node.token)
    }

    pub fn token_subtrie_ranges(&self) -> impl Iterator<Item = GpuTrieNode> {
        let mut stack: Vec<usize> = Vec::new();
        self.tokens.iter().enumerate().map(move |(index, n)| {
            let (start, end) = n.subtrie_range;

            stack.truncate(n.height);
            let parent = stack.last().map(|i| *i as i32).unwrap_or(-1);
            stack.push(index);

            GpuTrieNode {
                trie_start: start as u32,
                trie_end: end as u32,
                height: n.height as u32,
                parent,
            }
        })
    }

    pub fn token_seeds(&self) -> impl Iterator<Item = u64> {
        self.tokens.iter().map(|n| n.node.seed)
    }

    // pub fn fill_bitmasks(
    //     &self,
    //     bitmasks: &mut [u32],
    //     vocab_size: usize,
    //     grammar: &mut Grammar,
    // ) -> bool {
    //     let tokens: Box<[(u64, u32)]> =
    //         self.tokens.iter().map(|token| (token.node.token, token.height as u32)).collect();
    //     fill_bitmasks_flat(&tokens, bitmasks, vocab_size, grammar)
    // }

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
        mut grammar: Option<&mut Grammar>,
    ) -> Result<Box<[(usize, u64, u64)]>, TrieAcceptError> {
        let mut current_token = self.root().unwrap();
        let mut accepted = Vec::new();
        loop {
            let current_token_index = self.index(current_token).unwrap();
            let current_token_id = sampled_tokens[current_token_index];

            accepted.push((current_token_index, current_token.token, current_token_id));
            if let Some(grammar) = grammar.as_mut()
                && !grammar.is_terminated()
            {
                grammar.accept_token(current_token_id)?;
            }

            let Some(next_token) = current_token.get(current_token_id) else {
                break;
            };

            // The grammar may already be terminated here: when the draft model
            // proposed the grammar-closing token the trie descends one node
            // further, where the mask only allows stop tokens.
            current_token = next_token;
        }

        Ok(accepted.into_boxed_slice())
    }
}

pub fn fill_bitmasks(
    nodes: &[GpuTrieNode],
    tokens: &[u64],
    bitmasks: &mut [u32],
    vocab_size: usize,
    grammar: &mut Grammar,
) -> bool {
    let vocab_size_in_u32s = vocab_size.div_ceil(DataType::U32.size_in_bits());
    assert!(bitmasks.len() == nodes.len() * vocab_size_in_u32s);

    let mut any_non_full = false;
    let mut last_token_height = 0usize;
    for ((token_index, &node), bitmask) in
        nodes.iter().enumerate().zip_eq(bitmasks.chunks_exact_mut(vocab_size_in_u32s))
    {
        let height = node.height as usize;
        if token_index > 0 {
            if height <= last_token_height {
                grammar.rollback(last_token_height - height + 1);
            }
            grammar.accept_token(tokens[node.trie_start as usize]).expect("flat trie doesn't match grammar");
        }

        any_non_full |= grammar.next_bitmask(bitmask);

        last_token_height = height;
    }

    if last_token_height > 0 {
        grammar.rollback(last_token_height);
    }

    any_non_full
}
pub fn accept(
    nodes: &[GpuTrieNode],
    output_tokens: &[u64],
    input_tokens: &[u32],
    mut grammar: Option<&mut Grammar>,
) -> Result<(Box<[u32]>, Box<[u64]>, Box<[u64]>), TrieAcceptError> {
    assert_eq!(nodes.len(), input_tokens.len());
    assert_eq!(nodes.len(), output_tokens.len());
    assert!(!nodes.is_empty(), "the trie needs at least a root");

    let mut accepted_indices = Vec::new();
    let mut accepted_input_token_ids = Vec::new();
    let mut accepted_output_token_ids = Vec::new();
    let mut current = 0usize;
    loop {
        let sampled = output_tokens[current];
        accepted_indices.push(current as u32);
        accepted_input_token_ids.push(input_tokens[current] as u64);
        accepted_output_token_ids.push(sampled);

        if let Some(grammar) = grammar.as_mut()
            && !grammar.is_terminated()
        {
            grammar.accept_token(sampled)?;
        }

        let end = nodes[current].trie_end as usize;
        let next = (current + 1..=end)
            .find(|&child| nodes[child].parent == current as i32 && input_tokens[child] as u64 == sampled);
        let Some(next) = next else {
            break;
        };
        current = next;
    }

    Ok((
        accepted_indices.into_boxed_slice(),
        accepted_input_token_ids.into_boxed_slice(),
        accepted_output_token_ids.into_boxed_slice(),
    ))
}

#[cfg(test)]
#[path = "../unit/trie_test.rs"]
mod tests;
