use std::{collections::VecDeque, sync::Arc, time::Duration};

use thiserror::Error;

use crate::{
    backends::common::{Backend, Context, Encoder, gpu_types::trie::TrieNode as GpuTrieNode},
    encodable_block::{batch_topology::BatchTopology, decoder::DecoderError, transformer::TransformerState},
    engine::language_model::LanguageModel,
};

const PREFILL_MAX_BATCH_SIZE: u32 = 1024;

#[derive(Debug, Error)]
pub enum LanguageModelForwardError<B: Backend> {
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("decoder error: {0}")]
    Decoder(#[from] DecoderError<B>),
}

#[derive(Debug, Clone, Copy)]
pub struct MockTree {
    pub max_width: u32,
    pub balanced: bool,
}

impl MockTree {
    pub fn linearize(
        &self,
        node_count: u32,
    ) -> (Box<[GpuTrieNode]>, Box<[u32]>) {
        assert!(self.max_width > 0, "mock tree needs a positive max width");
        assert!(node_count > 0, "mock tree needs at least one node");

        let node_count = node_count as usize;
        let mut children = vec![Vec::new(); node_count];
        if self.balanced {
            let mut expandable = VecDeque::from([0usize]);
            for node in 1..node_count {
                let parent = expandable.pop_front().unwrap();
                children[parent].push(node);
                if children[parent].len() < self.max_width as usize {
                    expandable.push_back(parent);
                }
                expandable.push_back(node);
            }
        } else {
            let mut parent = 0usize;
            for node in 1..node_count {
                children[parent].push(node);
                if children[parent].len() == self.max_width as usize {
                    parent = children[parent][0];
                }
            }
        }

        let mut longest = vec![1u32; node_count];
        for (node, node_children) in children.iter().enumerate().rev() {
            for &child in node_children {
                longest[node] = longest[node].max(longest[child] + 1);
            }
        }

        let mut flat_index = vec![0u32; node_count];
        fn visit(
            node: usize,
            height: u32,
            children: &[Vec<usize>],
            flat: &mut Vec<GpuTrieNode>,
            flat_index: &mut [u32],
        ) -> u32 {
            let start = flat.len() as u32;
            flat_index[node] = start;
            flat.push(GpuTrieNode {
                trie_start: start,
                trie_end: start,
                height,
            });
            for &child in &children[node] {
                flat[start as usize].trie_end = visit(child, height + 1, children, flat, flat_index);
            }
            flat[start as usize].trie_end
        }
        let mut flat = Vec::with_capacity(node_count);
        visit(0, 0, &children, &mut flat, &mut flat_index);

        let mut path = Vec::with_capacity(longest[0] as usize);
        let mut node = 0usize;
        loop {
            path.push(flat_index[node]);
            match children[node].iter().max_by_key(|&&child| longest[child]) {
                Some(&child) => node = child,
                None => break,
            }
        }

        (flat.into_boxed_slice(), path.into_boxed_slice())
    }
}

pub fn chain_nodes(token_count: u32) -> Box<[GpuTrieNode]> {
    (0..token_count)
        .map(|index| GpuTrieNode {
            trie_start: index,
            trie_end: token_count - 1,
            height: index,
        })
        .collect()
}

impl<B: Backend> LanguageModel<B> {
    pub fn forward(
        &self,
        prefix_length: u32,
        suffix_length: u32,
        suffix_speculative: Option<MockTree>,
    ) -> Result<Duration, LanguageModelForwardError<B>> {
        assert!(suffix_length > 0, "forward pass needs at least one suffix token");

        let context = &self.engine.context;
        // Mixer encode can stash pool allocations into the state until
        // encode_accept, so the pool must outlive `state`.
        let allocation_pool = Arc::new(context.create_allocation_pool(false));
        let mut state = self
            .decoder
            .create_empty_state(Some(prefix_length + suffix_length), context)
            .map_err(LanguageModelForwardError::Backend)?;

        if prefix_length > 0 {
            let number_of_batches = prefix_length.div_ceil(PREFILL_MAX_BATCH_SIZE);
            state
                .prepare(
                    state.context_length() + (number_of_batches - 1) * PREFILL_MAX_BATCH_SIZE,
                    prefix_length.min(PREFILL_MAX_BATCH_SIZE),
                    context,
                )
                .map_err(LanguageModelForwardError::Backend)?;

            let mut encoder =
                Encoder::<B>::new_with_pool_name(context, allocation_pool.clone(), Some("forward prefix"))
                    .map_err(LanguageModelForwardError::Backend)?;
            let mut remaining = prefix_length;
            while remaining > 0 {
                let chunk = remaining.min(PREFILL_MAX_BATCH_SIZE);
                self.encode_pass(&mut state, &chain_nodes(chunk), true, false, &mut encoder)?;
                state
                    .encode_accept(&(0..chunk).collect::<Box<[_]>>(), &mut encoder)
                    .map_err(LanguageModelForwardError::Backend)?;
                remaining -= chunk;
            }
            encoder.end_encoding().submit().wait_until_completed().map_err(LanguageModelForwardError::Backend)?;
        }

        let (suffix_nodes, accept_indices, full_accept) = match &suffix_speculative {
            Some(tree) => {
                let (nodes, longest_path) = tree.linearize(suffix_length);
                (nodes, longest_path, false)
            },
            None => (chain_nodes(suffix_length), (0..suffix_length).collect(), true),
        };

        let mut encoder = Encoder::<B>::new_with_pool_name(context, allocation_pool.clone(), Some("forward suffix"))
            .map_err(LanguageModelForwardError::Backend)?;
        state.prepare(state.context_length(), suffix_length, context).map_err(LanguageModelForwardError::Backend)?;
        self.encode_pass(&mut state, &suffix_nodes, full_accept, true, &mut encoder)?;
        state.encode_accept(&accept_indices, &mut encoder).map_err(LanguageModelForwardError::Backend)?;
        let completed =
            encoder.end_encoding().submit().wait_until_completed().map_err(LanguageModelForwardError::Backend)?;
        Ok(completed.gpu_execution_time())
    }

    fn encode_pass(
        &self,
        state: &mut TransformerState<B>,
        nodes: &[GpuTrieNode],
        full_accept: bool,
        with_logits: bool,
        encoder: &mut Encoder<B>,
    ) -> Result<(), LanguageModelForwardError<B>> {
        let token_count = nodes.len() as u32;
        let token_ids = encoder
            .allocate_constant_from_slice(&vec![0u32; token_count as usize])
            .map_err(LanguageModelForwardError::Backend)?;
        let batch_dim = BatchTopology::new(nodes, full_accept);

        self.decoder
            .encode(&token_ids, &batch_dim, with_logits.then_some(0..token_count), None, state, encoder)
            .map_err(LanguageModelForwardError::Decoder)?;

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/forward_test.rs"]
mod tests;
