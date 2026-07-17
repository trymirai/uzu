use std::{cmp::Ordering, collections::BinaryHeap, rc::Rc};

use thiserror::Error;

pub use crate::encodable_block::dflash::DFlashState;
use crate::{
    backends::common::{Allocation, AllocationType, Backend, Context, Encoder},
    config::speculator::dflash::DFlashSpeculatorConfig,
    data_type::DataType,
    encodable_block::{
        dflash::{DFlashChainOutput, DFlashDraft, DFlashEncodeError},
        embedding::{Embedding, EmbeddingError},
        weaver::{Weaver, WeaverBatchStepInput, WeaverEncodeError},
    },
};

#[derive(Clone, Copy, Debug)]
pub struct DFlashTreeOptions {
    pub budget: usize,
    pub frontier_width: usize,
    pub children_per_node: usize,
}

const MAX_TREE_BUDGET: usize = 4096;
const MAX_FRONTIER_WIDTH: usize = 8;
const MAX_CHILDREN_PER_NODE: usize = 8;

impl Default for DFlashTreeOptions {
    fn default() -> Self {
        Self {
            budget: 128,
            frontier_width: 8,
            children_per_node: 8,
        }
    }
}

pub struct SpeculationTree<B: Backend> {
    /// Token IDs in verifier-ready DFS order; slot 0 is the target bonus root.
    pub tokens: Allocation<B>,
    /// Parent slot for each DFS node (`-1` only for the root).
    pub parents: Allocation<B>,
    /// Absolute depth within the proposed tree.
    pub depths: Allocation<B>,
    /// Weaver log probability for the selected edge into each node.
    pub draft_logprobs: Allocation<B>,
    pub len: usize,
}

#[derive(Debug, Error)]
pub enum DFlashTreeError<B: Backend> {
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
    #[error("DFlash error: {0}")]
    DFlash(#[from] DFlashEncodeError<B>),
    #[error("Weaver error: {0}")]
    Weaver(#[from] WeaverEncodeError<B>),
    #[error("DFlash tree construction requires target hidden features in the state")]
    MissingTargetHidden,
    #[error("invalid tree options")]
    InvalidOptions,
}

struct HostTreeNode {
    token: u32,
    parent: Option<usize>,
    depth: usize,
    logprob: f32,
    score: f32,
    children: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct FrontierEntry {
    score: f32,
    parent: usize,
    token: u32,
    logprob: f32,
}

impl PartialEq for FrontierEntry {
    fn eq(
        &self,
        other: &Self,
    ) -> bool {
        self.score == other.score && self.parent == other.parent && self.token == other.token
    }
}

impl Eq for FrontierEntry {}

impl PartialOrd for FrontierEntry {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FrontierEntry {
    fn cmp(
        &self,
        other: &Self,
    ) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.parent.cmp(&self.parent))
            .then_with(|| other.token.cmp(&self.token))
    }
}

fn build_host_tree<E>(
    mut nodes: Vec<HostTreeNode>,
    budget: usize,
    frontier_width: usize,
    mut expand: impl FnMut(&[usize], &mut Vec<HostTreeNode>, &mut BinaryHeap<FrontierEntry>) -> Result<(), E>,
) -> Result<Vec<HostTreeNode>, E> {
    let mut frontier = BinaryHeap::new();
    expand(&[0], &mut nodes, &mut frontier)?;
    while nodes.len() < budget + 1 && !frontier.is_empty() {
        let remaining = budget + 1 - nodes.len();
        let mut expanded = Vec::with_capacity(frontier_width.min(remaining));
        for _ in 0..frontier_width.min(remaining) {
            let Some(entry) = frontier.pop() else {
                break;
            };
            let parent_index = entry.parent;
            let index = nodes.len();
            nodes.push(HostTreeNode {
                token: entry.token,
                parent: Some(parent_index),
                depth: nodes[parent_index].depth + 1,
                logprob: entry.logprob,
                score: entry.score,
                children: Vec::new(),
            });
            nodes[parent_index].children.push(index);
            expanded.push(index);
        }
        expand(&expanded, &mut nodes, &mut frontier)?;
    }
    Ok(nodes)
}

pub struct DFlashSpeculator<B: Backend> {
    pub(crate) context: Rc<B::Context>,
    pub(crate) model: DFlashDraft<B>,
    pub(crate) weaver: Option<Weaver<B>>,
    pub(crate) config: DFlashSpeculatorConfig,
}

/// Unified DFlash + Weaver drafting facade used by the decoder seam.
///
/// The target model owns the embedding/readout weights, so the decoder binds
/// that immutable view once instead of making the speculator load a second
/// copy of the target vocabulary.
pub struct DFlashWeaver<'a, B: Backend> {
    speculator: &'a DFlashSpeculator<B>,
    target_embedding: &'a Embedding<B>,
}

impl<B: Backend> DFlashSpeculator<B> {
    pub(crate) const fn new(
        context: Rc<B::Context>,
        model: DFlashDraft<B>,
        weaver: Option<Weaver<B>>,
        config: DFlashSpeculatorConfig,
    ) -> Self {
        Self {
            context,
            model,
            weaver,
            config,
        }
    }

    pub(crate) fn empty_state(
        &self,
        context_capacity: usize,
    ) -> Result<DFlashState<B>, B::Error> {
        self.model.empty_state(context_capacity, &self.context)
    }

    pub(crate) fn bind_target_embedding<'a>(
        &'a self,
        target_embedding: &'a Embedding<B>,
    ) -> DFlashWeaver<'a, B> {
        DFlashWeaver {
            speculator: self,
            target_embedding,
        }
    }

    /// Appends target-model auxiliary layers and final normalized hidden
    /// state. The input allocations must remain alive until `encoder`
    /// completes because the copies are queued asynchronously.
    pub(crate) fn append_state(
        &self,
        state: &mut DFlashState<B>,
        target_features: &[Allocation<B>],
        target_hidden: &Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.model.append_state(state, target_features, target_hidden, encoder)
    }

    /// Appends accepted, newly committed rows, including the bonus root when
    /// it was not already in the DFlash context. The final hidden allocation
    /// must contain the same rows in the same order.
    pub(crate) fn update_state(
        &self,
        state: &mut DFlashState<B>,
        accepted_target_features: &[Allocation<B>],
        accepted_target_hidden: &Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.append_state(state, accepted_target_features, accepted_target_hidden, encoder)
    }

    pub(crate) fn encode_chain(
        &self,
        state: &mut DFlashState<B>,
        noise_token_ids: &Allocation<B>,
        target_embedding: &Embedding<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<DFlashChainOutput<B>, DFlashEncodeError<B>> {
        self.model.encode_chain(state, noise_token_ids, target_embedding, encoder)
    }

    fn encode_tree_with_embedding(
        &self,
        state: &mut DFlashState<B>,
        bonus_token: u32,
        target_embedding: &Embedding<B>,
        options: DFlashTreeOptions,
    ) -> Result<SpeculationTree<B>, DFlashTreeError<B>> {
        // `append_state` queues work on the caller's encoder.  Its target
        // feature allocation must be complete before this host-side copyout;
        // the decoder seam therefore submits/waits that encoder between the
        // append and this method.
        if options.budget == 0
            || options.budget > MAX_TREE_BUDGET
            || options.frontier_width == 0
            || options.frontier_width > MAX_FRONTIER_WIDTH
            || options.children_per_node == 0
            || options.children_per_node > MAX_CHILDREN_PER_NODE
        {
            return Err(DFlashTreeError::InvalidOptions);
        }
        let block_size = self.block_size();
        let target_model_dim = self.config.draft_config.model_dim;
        let vocab_size = self.config.draft_config.vocab_size;
        let pool_size =
            self.config.weaver_config.as_ref().map_or(1, |config| config.candidate_pool_size.min(vocab_size));
        if target_embedding.vocab_size() != self.config.draft_config.vocab_size
            || target_embedding.model_dim() != target_model_dim
            || self.config.weaver_config.as_ref().is_some_and(|config| config.target_model_dim != target_model_dim)
        {
            return Err(DFlashTreeError::InvalidOptions);
        }
        let target_hidden_size = state.last_target_hidden().ok_or(DFlashTreeError::MissingTargetHidden)?.size();
        if target_hidden_size != target_model_dim * DataType::BF16.size_in_bytes() {
            return Err(DFlashTreeError::InvalidOptions);
        }

        let mut encoder = Encoder::new(&*self.context).map_err(DFlashTreeError::Backend)?;
        let mut noise_ids =
            encoder.allocate_constant(block_size * DataType::U64.size_in_bytes()).map_err(DFlashTreeError::Backend)?;
        let mut noise = vec![self.config.draft_config.mask_token_id; block_size];
        noise[0] = bonus_token as u64;
        noise_ids.copyin(&noise);
        let chain = self.encode_chain(state, &noise_ids, target_embedding, &mut encoder)?;
        let (candidate_ids_allocation, candidate_scores_allocation) = self
            .model
            .encode_top_k(&chain.logits, block_size - 1, vocab_size, pool_size, &mut encoder)
            .map_err(DFlashTreeError::Backend)?;
        let completed = encoder.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
        let candidate_ids = candidate_ids_allocation.copyout::<u32>();
        let candidate_scores = candidate_scores_allocation.copyout::<f32>();
        drop(candidate_ids_allocation);
        drop(candidate_scores_allocation);
        let chain_logits = self.weaver.is_none().then(|| chain.logits.copyout::<f32>());
        drop(noise_ids);
        let candidate_pools = (0..block_size - 1)
            .map(|row| {
                (0..pool_size)
                    .map(|rank| {
                        let index = row * pool_size + rank;
                        (candidate_ids[index], candidate_scores[index])
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let mut nodes = vec![HostTreeNode {
            token: bonus_token,
            parent: None,
            depth: 0,
            logprob: 0.0,
            score: 0.0,
            children: Vec::new(),
        }];
        let Some(weaver) = self.weaver.as_ref() else {
            let logits = chain_logits.as_ref().unwrap();
            for depth in 0..options.budget.min(block_size.saturating_sub(1)) {
                let (token, _) = candidate_pools[depth][0];
                let row_start = depth * vocab_size;
                let logprob = log_softmax(&logits[row_start..row_start + vocab_size])[token as usize];
                let parent = nodes.len() - 1;
                let index = nodes.len();
                nodes.push(HostTreeNode {
                    token,
                    parent: Some(parent),
                    depth: depth + 1,
                    logprob,
                    score: nodes[parent].score + logprob,
                    children: Vec::new(),
                });
                nodes[parent].children.push(index);
            }
            let tree = Self::finish_tree(&self.context, nodes)?;
            drop(chain);
            drop(completed);
            return Ok(tree);
        };

        let max_depth = self.config.weaver_config.as_ref().unwrap().max_depth;
        let lookahead_count = max_depth.min(block_size.saturating_sub(1));
        let target_hidden = state.last_target_hidden().ok_or(DFlashTreeError::MissingTargetHidden)?;
        let prefix = weaver.prompt_prefix(target_hidden, &chain.hidden, 1, lookahead_count, &self.context)?;
        drop(chain);
        drop(completed);
        let mut weaver_state = weaver.create_node_state(options.budget + 1, &self.context)?;
        // Expand the root once to seed the frontier.  Thereafter the heap
        // contains candidate edges, not already-materialized tree nodes: each
        // iteration pops the globally best `frontier_width` edges, materializes
        // those nodes, and expands only those winners.
        let expand = |parent_indices: &[usize],
                      nodes: &mut Vec<HostTreeNode>,
                      frontier: &mut BinaryHeap<FrontierEntry>|
         -> Result<(), DFlashTreeError<B>> {
            let parent_indices = parent_indices
                .iter()
                .copied()
                .filter(|&index| nodes[index].depth < lookahead_count && nodes[index].depth < max_depth)
                .collect::<Vec<_>>();
            if parent_indices.is_empty() {
                return Ok(());
            }
            let candidate_ids = parent_indices
                .iter()
                .map(|&index| candidate_pools[nodes[index].depth].iter().map(|&(token, _)| token).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let candidate_scores = parent_indices
                .iter()
                .map(|&index| candidate_pools[nodes[index].depth].iter().map(|&(_, score)| score).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let ancestor_indices = parent_indices
                .iter()
                .map(|&parent_index| {
                    let mut indices = Vec::new();
                    let mut ancestor = nodes[parent_index].parent;
                    while let Some(index) = ancestor {
                        indices.push(index);
                        ancestor = nodes[index].parent;
                    }
                    indices.reverse();
                    indices
                })
                .collect::<Vec<_>>();
            let steps = {
                let inputs = parent_indices
                    .iter()
                    .enumerate()
                    .map(|(batch_index, &parent_index)| WeaverBatchStepInput {
                        node_index: parent_index,
                        parent_token: nodes[parent_index].token,
                        candidates: &candidate_ids[batch_index],
                        candidate_scores: &candidate_scores[batch_index],
                        ancestors: &ancestor_indices[batch_index],
                        depth: nodes[parent_index].depth,
                    })
                    .collect::<Vec<_>>();
                weaver.step_batch(
                    &prefix,
                    &inputs,
                    &mut weaver_state,
                    options.children_per_node,
                    target_embedding,
                    &self.context,
                )?
            };
            for (parent_index, step) in parent_indices.into_iter().zip(steps) {
                for (token, logprob) in step.children {
                    let score = nodes[parent_index].score + logprob;
                    frontier.push(FrontierEntry {
                        score,
                        parent: parent_index,
                        token,
                        logprob,
                    });
                }
            }
            Ok(())
        };

        let nodes = build_host_tree(nodes, options.budget, options.frontier_width, expand)?;
        Self::finish_tree(&self.context, nodes)
    }

    fn finish_tree(
        context: &B::Context,
        nodes: Vec<HostTreeNode>,
    ) -> Result<SpeculationTree<B>, DFlashTreeError<B>> {
        let mut order = Vec::with_capacity(nodes.len());
        let mut stack = vec![0usize];
        while let Some(index) = stack.pop() {
            order.push(index);
            for &child in nodes[index].children.iter().rev() {
                stack.push(child);
            }
        }
        let mut expansion_to_dfs = vec![u32::MAX; order.len()];
        for (dfs, &expansion) in order.iter().enumerate() {
            expansion_to_dfs[expansion] = dfs as u32;
        }
        let mut tokens = vec![0u32; order.len()];
        let mut parents = vec![-1i32; order.len()];
        let mut depths = vec![0u32; order.len()];
        let mut draft_logprobs = vec![0.0f32; order.len()];
        for (dfs, &index) in order.iter().enumerate() {
            tokens[dfs] = nodes[index].token;
            parents[dfs] = nodes[index].parent.map_or(-1, |parent| expansion_to_dfs[parent] as i32);
            depths[dfs] = nodes[index].depth as u32;
            draft_logprobs[dfs] = nodes[index].logprob;
        }
        let tokens = global_allocation(context, &tokens).map_err(DFlashTreeError::Backend)?;
        let parents = global_allocation(context, &parents).map_err(DFlashTreeError::Backend)?;
        let depths = global_allocation(context, &depths).map_err(DFlashTreeError::Backend)?;
        let draft_logprobs = global_allocation(context, &draft_logprobs).map_err(DFlashTreeError::Backend)?;
        Ok(SpeculationTree {
            tokens,
            parents,
            depths,
            draft_logprobs,
            len: order.len(),
        })
    }

    pub(crate) fn block_size(&self) -> usize {
        self.model.block_size()
    }
}

impl<B: Backend> DFlashWeaver<'_, B> {
    pub fn empty_state(
        &self,
        context_capacity: usize,
    ) -> Result<DFlashState<B>, B::Error> {
        self.speculator.empty_state(context_capacity)
    }

    pub fn append_state(
        &self,
        state: &mut DFlashState<B>,
        target_features: &[Allocation<B>],
        target_hidden: &Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.speculator.append_state(state, target_features, target_hidden, encoder)
    }

    pub fn update_state(
        &self,
        state: &mut DFlashState<B>,
        accepted_target_features: &[Allocation<B>],
        accepted_target_hidden: &Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.speculator.update_state(state, accepted_target_features, accepted_target_hidden, encoder)
    }

    /// Builds one verifier-ready tree after the encoder that supplied the
    /// latest append/update has completed.
    ///
    /// `bonus_token` is the target model's next token at the current DFlash
    /// context boundary; it becomes slot zero and is verified with the tree.
    pub fn encode_tree(
        &self,
        state: &mut DFlashState<B>,
        bonus_token: u32,
        options: DFlashTreeOptions,
    ) -> Result<SpeculationTree<B>, DFlashTreeError<B>> {
        self.speculator.encode_tree_with_embedding(state, bonus_token, self.target_embedding, options)
    }
}

fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits.iter().map(|value| (value - max).exp()).sum::<f32>().ln() + max;
    logits.iter().map(|value| value - log_sum).collect()
}

fn global_allocation<B: Backend, T: bytemuck::NoUninit + bytemuck::AnyBitPattern>(
    context: &B::Context,
    values: &[T],
) -> Result<Allocation<B>, B::Error> {
    let mut allocation = context.create_allocation(std::mem::size_of_val(values), AllocationType::Global)?;
    allocation.copyin(values);
    Ok(allocation)
}
