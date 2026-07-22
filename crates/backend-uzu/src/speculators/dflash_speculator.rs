use std::{cmp::Ordering, collections::BinaryHeap, sync::Arc};

use thiserror::Error;

pub use crate::encodable_block::dflash::DFlashState;
use crate::{
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder, Kernels,
        kernel::{
            WeaverCandidateGatherKernel, WeaverFrontierScatterKernel, WeaverFrontierSelectKernel,
            weaver::{
                MAX_FRONTIER_SLOTS, NO_WINNER, TREE_LANE_CUM, TREE_LANE_DEPTH, TREE_LANE_LOGPROB, TREE_LANE_MASK,
                TREE_LANE_PARENT, TREE_LANE_TOKEN,
            },
        },
    },
    config::speculator::dflash::DFlashSpeculatorConfig,
    data_type::DataType,
    encodable_block::{
        dflash::DFlashDraft,
        embedding::{Embedding, EmbeddingError},
        weaver::{Weaver, WeaverBatchStepInput, WeaverEncodeError, WeaverGpuStepInput, WeaverNodeState, WeaverPrefix},
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
const MAX_TREE_FRONTIER_WIDTH: usize = 8;
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

struct DFlashChainOutput<B: Backend> {
    pool_ids: Allocation<B>,
    pool_scores: Allocation<B>,
    draft_logits: Allocation<B>,
    draft_hidden: Allocation<B>,
}

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
        nodes: &[HostTreeNode],
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
        if options.budget == 0
            || options.budget > MAX_TREE_BUDGET
            || options.frontier_width == 0
            || options.frontier_width > MAX_TREE_FRONTIER_WIDTH
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

        let mut encoder = Encoder::new(&*self.context).map_err(DFlashTreeError::Backend)?;
        let DFlashChainOutput {
            pool_ids,
            pool_scores,
            draft_logits,
            draft_hidden,
        } = self.encode_dflash_chain(
            &mut encoder,
            state,
            bonus_token,
            target_embedding,
            block_size,
            target_model_dim,
            pool_size,
        )?;

        let weaver = if let Some(weaver) = self.weaver.as_ref() {
            let max_depth = self.config.weaver_config.as_ref().expect("a Weaver implies a Weaver config").max_depth;
            let lookahead_count = max_depth.min(block_size.saturating_sub(1));
            let target_hidden = state.target_output_norm().ok_or(DFlashTreeError::MissingTargetHidden)?;
            let prefix = weaver.build_prefix(target_hidden, &draft_hidden, 1, lookahead_count, &mut encoder)?;
            Some((weaver, prefix, max_depth, lookahead_count))
        } else {
            None
        };

        if let Some((weaver, prefix, max_depth, lookahead_count)) = weaver.as_ref()
            && std::env::var("UZU_DFLASH_GPU_TREE").is_ok_and(|value| value == "1")
            && (options.budget + 1) * options.children_per_node <= MAX_FRONTIER_SLOTS
            // Fixed-width rounds may only pad the terminal round.
            && options.children_per_node.is_multiple_of(options.frontier_width)
            && *lookahead_count > 0
        {
            drop(draft_hidden);
            drop(draft_logits);
            let mut weaver_state = weaver.create_node_state(options.budget + 1, &self.context)?;
            let params = GpuTreeParams {
                weaver,
                prefix,
                target_embedding,
                pool_ids: &pool_ids,
                pool_scores: &pool_scores,
                pool_rows: block_size - 1,
                pool_size,
                options,
                max_depth: *max_depth,
                lookahead_count: *lookahead_count,
                bonus_token,
            };
            let tree = self.encode_gpu_tree(&mut encoder, &params, &mut weaver_state)?;
            let completed = encoder.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
            let slots = tree.copyout::<u32>();
            drop(tree);
            drop(pool_ids);
            drop(pool_scores);
            drop(completed);
            return Self::finish_tree(&self.context, tree_from_slots(&slots));
        }

        let completed = encoder.end_encoding().submit().wait_until_completed().map_err(DFlashTreeError::Backend)?;
        let pool_id_values = pool_ids.copyout::<u32>();
        let pool_score_values = pool_scores.copyout::<f32>();
        let candidate_pools = pool_id_values
            .chunks_exact(pool_size)
            .zip(pool_score_values.chunks_exact(pool_size))
            .map(|(ids, scores)| ids.iter().copied().zip(scores.iter().copied()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let host_logits = self.weaver.is_none().then(|| draft_logits.copyout::<f32>());
        drop(pool_ids);
        drop(pool_scores);
        drop(draft_logits);
        drop(draft_hidden);
        drop(completed);

        let mut nodes = vec![HostTreeNode {
            token: bonus_token,
            parent: None,
            depth: 0,
            logprob: 0.0,
            score: 0.0,
            children: Vec::new(),
        }];
        let Some((weaver, prefix, max_depth, lookahead_count)) = weaver.as_ref() else {
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
            return Self::finish_tree(&self.context, nodes);
        };

        let mut weaver_state = weaver.create_node_state(options.budget + 1, &self.context)?;
        let mut expansion = WeaverExpansion {
            speculator: self,
            weaver,
            prefix,
            weaver_state: &mut weaver_state,
            target_embedding,
            candidate_pools: &candidate_pools,
            children_per_node: options.children_per_node,
            lookahead_count: *lookahead_count,
            max_depth: *max_depth,
        };

        let nodes = build_host_tree(nodes, options.budget, options.frontier_width, |parents, nodes, frontier| {
            expansion.expand_round(parents, nodes, frontier)
        })?;
        Self::finish_tree(&self.context, nodes)
    }

    fn encode_dflash_chain(
        &self,
        encoder: &mut Encoder<B>,
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

        let mut noise_ids =
            encoder.allocate_constant(block_size * DataType::U32.size_in_bytes()).map_err(DFlashTreeError::Backend)?;
        let mut noise = vec![self.config.draft_config.mask_token_id as u32; block_size];
        noise[0] = bonus_token;
        noise_ids.copyin(&noise);
        let token_embeddings = target_embedding.encode_lookup(&noise_ids, block_size, encoder)?;
        let draft_hidden =
            self.model.encode_block(state, token_embeddings, encoder).map_err(DFlashTreeError::Backend)?;
        // The first block row is the bonus token; only the lookahead rows are ranked.
        let row_bytes = target_embedding.model_dim() * DataType::BF16.size_in_bytes();
        let mut lookahead_hidden =
            encoder.allocate_scratch((block_size - 1) * row_bytes).map_err(DFlashTreeError::Backend)?;
        encoder.encode_copy(&draft_hidden, row_bytes..block_size * row_bytes, &mut lookahead_hidden, ..);
        let draft_logits =
            target_embedding.encode_readout(block_size - 1, &lookahead_hidden, DataType::F32, encoder)?;
        let (pool_ids, pool_scores) = self
            .model
            .encode_top_k(&draft_logits, block_size - 1, pool_size, encoder)
            .map_err(DFlashTreeError::Backend)?;
        drop(noise_ids);
        drop(lookahead_hidden);
        Ok(DFlashChainOutput {
            pool_ids,
            pool_scores,
            draft_logits,
            draft_hidden,
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
        let tokens = global_allocation::<B, _>(context, &tokens).map_err(DFlashTreeError::Backend)?;
        let parents = global_allocation::<B, _>(context, &parents).map_err(DFlashTreeError::Backend)?;
        let depths = global_allocation::<B, _>(context, &depths).map_err(DFlashTreeError::Backend)?;
        let draft_logprobs = global_allocation::<B, _>(context, &draft_logprobs).map_err(DFlashTreeError::Backend)?;
        Ok(TreeProposal {
            token_ids: tokens,
            parents,
            depths,
            draft_logprobs,
            length: order.len(),
        })
    }

    fn encode_gpu_tree(
        &self,
        encoder: &mut Encoder<B>,
        params: &GpuTreeParams<'_, B>,
        state: &mut WeaverNodeState<B>,
    ) -> Result<Allocation<B>, DFlashTreeError<B>> {
        // A round only expands nodes materialized by earlier rounds.
        let context = &*self.context;
        let slots = params.options.budget + 1;
        let fanout = params.options.children_per_node;
        let capacity = slots * fanout;
        let width = params.options.frontier_width;
        let stride = params.max_depth;
        let pool_size = params.pool_size;

        let select =
            <B::Kernels as Kernels>::WeaverFrontierSelectKernel::new(context).map_err(DFlashTreeError::Backend)?;
        let gather =
            <B::Kernels as Kernels>::WeaverCandidateGatherKernel::new(context).map_err(DFlashTreeError::Backend)?;
        let scatter =
            <B::Kernels as Kernels>::WeaverFrontierScatterKernel::new(context).map_err(DFlashTreeError::Backend)?;

        let mut tree_values = vec![0u32; TREE_LANES * slots];
        for slot in 0..slots {
            tree_values[TREE_LANE_PARENT * slots + slot] = NO_WINNER;
        }
        tree_values[TREE_LANE_TOKEN * slots] = params.bonus_token;
        tree_values[TREE_LANE_MASK * slots] = 1;

        let mut tree = encoder.allocate_constant_from_slice(&tree_values).map_err(DFlashTreeError::Backend)?;
        let mut frontier = encoder
            .allocate_constant_from_slice(&vec![0u32; FRONTIER_LANES * capacity])
            .map_err(DFlashTreeError::Backend)?;
        let mut slot_ancestors =
            encoder.allocate_constant_from_slice(&vec![0u32; slots * stride]).map_err(DFlashTreeError::Backend)?;

        let mut round_token_values = vec![0u32; width];
        round_token_values[0] = params.bonus_token;
        let mut round_valid_values = vec![0u32; width];
        round_valid_values[0] = 1;
        let mut round_token =
            encoder.allocate_constant_from_slice(&round_token_values).map_err(DFlashTreeError::Backend)?;
        let mut round_metadata =
            encoder.allocate_constant_from_slice(&vec![0u32; 3 * width]).map_err(DFlashTreeError::Backend)?;
        let mut round_ancestors =
            encoder.allocate_constant_from_slice(&vec![0u32; width * stride]).map_err(DFlashTreeError::Backend)?;
        let mut round_valid =
            encoder.allocate_constant_from_slice(&round_valid_values).map_err(DFlashTreeError::Backend)?;
        let mut round_candidate_ids =
            encoder.allocate_constant_from_slice(&vec![0u32; width * pool_size]).map_err(DFlashTreeError::Backend)?;
        let mut round_candidate_scores =
            encoder.allocate_constant_from_slice(&vec![0.0f32; width * pool_size]).map_err(DFlashTreeError::Backend)?;

        let mut slot_start = 0;
        while slot_start < slots {
            let rows = if slot_start == 0 {
                1
            } else {
                width.min(slots - slot_start)
            };
            if slot_start > 0 {
                select.encode(
                    &mut frontier,
                    &mut tree,
                    &mut slot_ancestors,
                    &mut round_token,
                    &mut round_metadata,
                    &mut round_ancestors,
                    &mut round_valid,
                    capacity as u32,
                    slots as u32,
                    rows as u32,
                    slot_start as u32,
                    stride as u32,
                    params.max_depth as u32,
                    params.lookahead_count as u32,
                    encoder,
                );
                gather.encode(
                    params.pool_ids,
                    params.pool_scores,
                    &round_metadata,
                    &mut round_candidate_ids,
                    &mut round_candidate_scores,
                    rows as u32,
                    params.pool_rows as u32,
                    pool_size as u32,
                    encoder,
                );
            }
            let (candidate_ids, candidate_scores) = if slot_start == 0 {
                (params.pool_ids, params.pool_scores)
            } else {
                (&round_candidate_ids, &round_candidate_scores)
            };
            let input = WeaverGpuStepInput {
                rows,
                candidates: pool_size,
                ancestor_stride: stride,
                token_ids: &round_token,
                candidate_ids,
                candidate_scores,
                ancestor_indices: &round_ancestors,
                metadata: &round_metadata,
            };
            let (child_ids, child_logprobs) = params.weaver.encode_step_batch_gpu(
                params.prefix,
                &input,
                state,
                fanout,
                params.target_embedding,
                encoder,
            )?;
            scatter.encode(
                &tree,
                &round_metadata,
                &round_valid,
                &child_ids,
                &child_logprobs,
                &mut frontier,
                capacity as u32,
                slots as u32,
                rows as u32,
                fanout as u32,
                encoder,
            );
            drop(child_ids);
            drop(child_logprobs);
            slot_start += rows;
        }
        Ok(tree)
    }
}

const TREE_LANES: usize = 6;
const FRONTIER_LANES: usize = 7;
fn tree_from_slots(tree: &[u32]) -> Vec<HostTreeNode> {
    assert!(tree.len().is_multiple_of(TREE_LANES), "tree array must be {TREE_LANES} equal-length lanes");
    let slots = tree.len() / TREE_LANES;
    let lane = |lane: usize, slot: usize| tree[lane * slots + slot];
    let mut slot_to_node = vec![usize::MAX; slots];
    let mut nodes: Vec<HostTreeNode> = Vec::with_capacity(slots);
    for slot in 0..slots {
        if lane(TREE_LANE_MASK, slot) == 0 {
            continue;
        }
        let parent_slot = lane(TREE_LANE_PARENT, slot) as i32;
        let parent = (parent_slot >= 0).then(|| {
            let parent = slot_to_node[parent_slot as usize];
            assert_ne!(parent, usize::MAX, "tree slot {slot} names padding slot {parent_slot} as its parent");
            parent
        });
        let index = nodes.len();
        slot_to_node[slot] = index;
        if let Some(parent) = parent {
            nodes[parent].children.push(index);
        }
        nodes.push(HostTreeNode {
            token: lane(TREE_LANE_TOKEN, slot),
            parent,
            depth: lane(TREE_LANE_DEPTH, slot) as usize,
            logprob: f32::from_bits(lane(TREE_LANE_LOGPROB, slot)),
            score: f32::from_bits(lane(TREE_LANE_CUM, slot)),
            children: Vec::new(),
        });
    }
    nodes
}

struct GpuTreeParams<'a, B: Backend> {
    weaver: &'a Weaver<B>,
    prefix: &'a WeaverPrefix<B>,
    target_embedding: &'a Embedding<B>,
    pool_ids: &'a Allocation<B>,
    pool_scores: &'a Allocation<B>,
    pool_rows: usize,
    pool_size: usize,
    options: DFlashTreeOptions,
    max_depth: usize,
    lookahead_count: usize,
    bonus_token: u32,
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
