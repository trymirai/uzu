use std::{cmp::Ordering, collections::BinaryHeap, sync::Arc};

use thiserror::Error;

pub use crate::encodable_block::dflash::DFlashState;
use crate::{
    backends::common::{Allocation, AllocationType, Backend, Completed, Context, Encoder},
    config::speculator::dflash::DFlashSpeculatorConfig,
    data_type::DataType,
    encodable_block::{
        dflash::DFlashDraft,
        embedding::{Embedding, EmbeddingError},
        weaver::{Weaver, WeaverBatchStepInput, WeaverEncodeError, WeaverNodeState, WeaverPrefix},
    },
    engine::language_model::LanguageModel,
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

pub struct TreeProposal<B: Backend> {
    /// DFS order; slot 0 is the bonus root.
    pub token_ids: Allocation<B>,
    pub parents: Allocation<B>,
    pub depths: Allocation<B>,
    pub draft_logprobs: Allocation<B>,
    pub length: usize,
}

#[derive(Debug, Error)]
pub enum DFlashTreeError<B: Backend> {
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
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
    mut expand: impl FnMut(&[usize], &mut [HostTreeNode], &mut BinaryHeap<FrontierEntry>) -> Result<(), E>,
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

/// GPU-produced draft data pulled back to the host by the DFlash chain: one
/// candidate `(token, score)` pool per lookahead row, the full logits (only
/// materialized when Weaver is absent), and the draft-hidden / completion
/// handles the caller keeps alive until Weaver's prefix build is done.
struct DFlashChainOutput<B: Backend> {
    candidate_pools: Vec<Vec<(u32, f32)>>,
    host_logits: Option<Vec<f32>>,
    draft_hidden: Allocation<B>,
    completed: Completed<B>,
}

/// Per-round driver for the best-first frontier loop: assembles one Weaver
/// batch from the round's winners, runs it, and pushes the resulting child
/// edges onto the frontier.
struct WeaverExpansion<'a, B: Backend> {
    speculator: &'a DFlashSpeculator<B>,
    weaver: &'a Weaver<B>,
    prefix: &'a WeaverPrefix<B>,
    weaver_state: &'a mut WeaverNodeState<B>,
    target_embedding: &'a Embedding<B>,
    candidate_pools: &'a [Vec<(u32, f32)>],
    children_per_node: usize,
    lookahead_count: usize,
    max_depth: usize,
}

impl<B: Backend> WeaverExpansion<'_, B> {
    fn expand_round(
        &mut self,
        parent_indices: &[usize],
        nodes: &mut [HostTreeNode],
        frontier: &mut BinaryHeap<FrontierEntry>,
    ) -> Result<(), DFlashTreeError<B>> {
        let parent_indices = parent_indices
            .iter()
            .copied()
            .filter(|&index| nodes[index].depth < self.lookahead_count && nodes[index].depth < self.max_depth)
            .collect::<Vec<_>>();
        if parent_indices.is_empty() {
            return Ok(());
        }
        let candidate_ids = parent_indices
            .iter()
            .map(|&index| self.candidate_pools[nodes[index].depth].iter().map(|&(token, _)| token).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let candidate_scores = parent_indices
            .iter()
            .map(|&index| self.candidate_pools[nodes[index].depth].iter().map(|&(_, score)| score).collect::<Vec<_>>())
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
            self.weaver.step_batch(
                self.prefix,
                &inputs,
                self.weaver_state,
                self.children_per_node,
                self.target_embedding,
                &self.speculator.context,
            )?
        };
        for (parent_index, children) in parent_indices.into_iter().zip(steps) {
            for (token, logprob) in children {
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
    }
}

pub struct DFlashSpeculator<B: Backend> {
    pub(crate) context: Arc<B::Context>,
    pub(crate) model: DFlashDraft<B>,
    pub(crate) weaver: Option<Weaver<B>>,
    pub(crate) config: DFlashSpeculatorConfig,
}

/// DFlash-TfM bound to target embedding weights.
#[derive(Clone, Copy)]
pub struct DFlashTfm<'a, B: Backend> {
    speculator: &'a DFlashSpeculator<B>,
    target_embedding: &'a Embedding<B>,
}

impl<B: Backend> DFlashSpeculator<B> {
    pub(crate) const fn new(
        context: Arc<B::Context>,
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

    fn propose_tree_with_embedding(
        &self,
        state: &mut DFlashState<B>,
        bonus_token: u32,
        target_embedding: &Embedding<B>,
        options: DFlashTreeOptions,
    ) -> Result<TreeProposal<B>, DFlashTreeError<B>> {
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
        let block_size = self.model.block_size();
        let target_model_dim = self.config.draft_config.model_dim;
        let vocab_size = self.config.draft_config.vocab_size;
        let pool_size =
            self.config.weaver_config.as_ref().map_or(1, |config| config.candidate_pool_size.min(vocab_size));
        if bonus_token as usize >= vocab_size
            || pool_size == 0
            || options.children_per_node > pool_size
            || target_embedding.vocab_size() != self.config.draft_config.vocab_size
            || target_embedding.model_dim() != target_model_dim
            || self.config.weaver_config.as_ref().is_some_and(|config| config.target_model_dim != target_model_dim)
        {
            return Err(DFlashTreeError::InvalidOptions);
        }

        let DFlashChainOutput {
            candidate_pools,
            host_logits,
            draft_hidden,
            completed,
        } = self.encode_dflash_chain(state, bonus_token, target_embedding, block_size, target_model_dim, pool_size)?;

        let mut nodes = vec![HostTreeNode {
            token: bonus_token,
            parent: None,
            depth: 0,
            logprob: 0.0,
            score: 0.0,
            children: Vec::new(),
        }];
        let Some(weaver) = self.weaver.as_ref() else {
            let logits = host_logits.as_ref().unwrap();
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
            drop(draft_hidden);
            drop(completed);
            return Ok(tree);
        };

        let max_depth = self.config.weaver_config.as_ref().unwrap().max_depth;
        let lookahead_count = max_depth.min(block_size.saturating_sub(1));
        let target_hidden = state.target_output_norm().ok_or(DFlashTreeError::MissingTargetHidden)?;
        let prefix = weaver.build_prefix(target_hidden, &draft_hidden, 1, lookahead_count, &self.context)?;
        drop(draft_hidden);
        drop(completed);
        let mut weaver_state = weaver.create_node_state(options.budget + 1, &self.context)?;
        // Expand the root once to seed the frontier.  Thereafter the heap
        // contains candidate edges, not already-materialized tree nodes: each
        // iteration pops the globally best `frontier_width` edges, materializes
        // those nodes, and expands only those winners.
        let mut expansion = WeaverExpansion {
            speculator: self,
            weaver,
            prefix: &prefix,
            weaver_state: &mut weaver_state,
            target_embedding,
            candidate_pools: &candidate_pools,
            children_per_node: options.children_per_node,
            lookahead_count,
            max_depth,
        };

        let nodes = build_host_tree(nodes, options.budget, options.frontier_width, |parents, nodes, frontier| {
            expansion.expand_round(parents, nodes, frontier)
        })?;

        let tree = Self::finish_tree(&self.context, nodes)?;
        Ok(tree)
    }

    /// Runs the DFlash draft chain on the GPU and pulls its results back to the
    /// host: builds the noise block, encodes lookup -> draft transformer ->
    /// readout -> top-k in one command buffer, waits, copies out the candidate
    /// ids/scores (and full logits when Weaver is absent), and reshapes them
    /// into per-row candidate pools.  Records every timing this stage owns.
    fn encode_dflash_chain(
        &self,
        state: &mut DFlashState<B>,
        bonus_token: u32,
        target_embedding: &Embedding<B>,
        block_size: usize,
        target_model_dim: usize,
        pool_size: usize,
    ) -> Result<DFlashChainOutput<B>, DFlashTreeError<B>> {
        let target_hidden_size = state.target_output_norm().ok_or(DFlashTreeError::MissingTargetHidden)?.size();
        if target_hidden_size != target_model_dim * DataType::BF16.size_in_bytes() {
            return Err(DFlashTreeError::InvalidOptions);
        }

        let mut encoder = Encoder::new(&*self.context).map_err(DFlashTreeError::Backend)?;
        let mut noise_ids =
            encoder.allocate_constant(block_size * DataType::U32.size_in_bytes()).map_err(DFlashTreeError::Backend)?;
        let mut noise = vec![self.config.draft_config.mask_token_id as u32; block_size];
        noise[0] = bonus_token;
        noise_ids.copyin(&noise);
        let token_embeddings = target_embedding.encode_lookup(&noise_ids, block_size, &mut encoder)?;
        let draft_hidden =
            self.model.encode_block(state, token_embeddings, &mut encoder).map_err(DFlashTreeError::Backend)?;
        // The first block row is the bonus token; only the lookahead rows are ranked.
        let row_bytes = target_embedding.model_dim() * DataType::BF16.size_in_bytes();
        let mut lookahead_hidden =
            encoder.allocate_scratch((block_size - 1) * row_bytes).map_err(DFlashTreeError::Backend)?;
        encoder.encode_copy(&draft_hidden, row_bytes..block_size * row_bytes, &mut lookahead_hidden, ..);
        let draft_logits =
            target_embedding.encode_readout(block_size - 1, &lookahead_hidden, DataType::F32, &mut encoder)?;
        let (candidate_ids_allocation, candidate_scores_allocation) = self
            .model
            .encode_top_k(&draft_logits, block_size - 1, pool_size, &mut encoder)
            .map_err(DFlashTreeError::Backend)?;
        let completed = encoder.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
        let candidate_ids = candidate_ids_allocation.copyout::<u32>();
        let candidate_scores = candidate_scores_allocation.copyout::<f32>();
        drop(candidate_ids_allocation);
        drop(candidate_scores_allocation);
        let host_logits = self.weaver.is_none().then(|| draft_logits.copyout::<f32>());
        drop(draft_logits);
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
        Ok(DFlashChainOutput {
            candidate_pools,
            host_logits,
            draft_hidden,
            completed,
        })
    }

    fn finish_tree(
        context: &B::Context,
        nodes: Vec<HostTreeNode>,
    ) -> Result<TreeProposal<B>, DFlashTreeError<B>> {
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
        Ok(TreeProposal {
            token_ids: tokens,
            parents,
            depths,
            draft_logprobs,
            length: order.len(),
        })
    }
}

impl<'a, B: Backend> DFlashTfm<'a, B> {
    pub fn new(
        speculator: &'a DFlashSpeculator<B>,
        target: &'a LanguageModel<B>,
    ) -> Self {
        Self {
            speculator,
            target_embedding: target.embedding(),
        }
    }
}

impl<B: Backend> DFlashTfm<'_, B> {
    pub fn empty_state(
        &self,
        context_capacity: usize,
    ) -> Result<DFlashState<B>, B::Error> {
        self.speculator.model.empty_state(context_capacity, &self.speculator.context)
    }

    pub fn hidden_feature_layer_indices(&self) -> &[usize] {
        &self.speculator.config.draft_config.target_layer_ids
    }

    pub fn append_state(
        &self,
        state: &mut DFlashState<B>,
        target_features: &[Allocation<B>],
        num_tokens: usize,
        target_output_norm: &Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        self.speculator.model.append_state(state, target_features, num_tokens, target_output_norm, encoder)
    }

    /// Proposes a tree rooted at `bonus_token`.
    /// The last append encoder must be complete.
    pub fn propose_tree(
        &self,
        state: &mut DFlashState<B>,
        bonus_token: u32,
        options: DFlashTreeOptions,
    ) -> Result<TreeProposal<B>, DFlashTreeError<B>> {
        self.speculator.propose_tree_with_embedding(state, bonus_token, self.target_embedding, options)
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
