use std::sync::Arc;

use thiserror::Error;

pub use crate::encodable_block::dflash::DFlashState;
use crate::{
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder, Kernels,
        gpu_types::weaver,
        kernel::{WeaverFrontierScatterKernel, WeaverFrontierSelectKernel},
    },
    config::speculator::dflash::DFlashSpeculatorConfig,
    data_type::DataType,
    encodable_block::{
        dflash::DFlashDraft,
        embedding::{Embedding, EmbeddingError},
        weaver::{Weaver, WeaverEncodeError, WeaverNodeState, WeaverPrefix, WeaverStepBatch},
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
    children: Vec<usize>,
}

struct DFlashChainOutput<B: Backend> {
    pool_ids: Allocation<B>,
    pool_scores: Allocation<B>,
    draft_logits: Allocation<B>,
    draft_hidden: Allocation<B>,
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

    fn propose_tree(
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

        if let Some(weaver) = self.weaver.as_ref() {
            let max_depth = self.config.weaver_config.as_ref().expect("a Weaver implies a Weaver config").max_depth;
            let lookahead_count = max_depth.min(block_size.saturating_sub(1));
            // The frontier holds one slot per (tree slot, child) pair; the select
            // kernel silently no-ops past its capacity.
            if (options.budget + 1) * options.children_per_node > weaver::FRONTIER_MAX_SLOTS || lookahead_count == 0 {
                return Err(DFlashTreeError::InvalidOptions);
            }
            let target_hidden = state.target_output_norm().ok_or(DFlashTreeError::MissingTargetHidden)?;
            let prefix = weaver.build_prefix(target_hidden, &draft_hidden, 1, lookahead_count, &mut encoder)?;
            drop(draft_hidden);
            drop(draft_logits);
            let mut weaver_state = weaver.create_node_state(options.budget + 1, &self.context)?;
            let arguments = TreeEncodingArguments {
                weaver,
                prefix: &prefix,
                target_embedding,
                pool_ids: &pool_ids,
                pool_scores: &pool_scores,
                pool_rows: block_size - 1,
                pool_size,
                options,
                max_depth,
                lookahead_count,
                bonus_token,
            };
            let tree = self.encode_tree(&mut encoder, &arguments, &mut weaver_state)?;
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
        let host_logits = draft_logits.copyout::<f32>();
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
            children: Vec::new(),
        }];
        for depth in 0..options.budget.min(block_size.saturating_sub(1)) {
            let token = pool_id_values[depth * pool_size];
            let row_start = depth * vocab_size;
            let logprob = log_softmax(&host_logits[row_start..row_start + vocab_size])[token as usize];
            let parent = nodes.len() - 1;
            let index = nodes.len();
            nodes.push(HostTreeNode {
                token,
                parent: Some(parent),
                depth: depth + 1,
                logprob,
                children: Vec::new(),
            });
            nodes[parent].children.push(index);
        }
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

    fn encode_tree(
        &self,
        encoder: &mut Encoder<B>,
        params: &TreeEncodingArguments<'_, B>,
        state: &mut WeaverNodeState<B>,
    ) -> Result<Allocation<B>, DFlashTreeError<B>> {
        let context = &*self.context;
        let slots = params.options.budget + 1;
        let fanout = params.options.children_per_node;
        let capacity = slots * fanout;
        let width = params.options.frontier_width;
        let stride = params.max_depth;
        let pool_size = params.pool_size;

        let select =
            <B::Kernels as Kernels>::WeaverFrontierSelectKernel::new(context).map_err(DFlashTreeError::Backend)?;
        let scatter =
            <B::Kernels as Kernels>::WeaverFrontierScatterKernel::new(context).map_err(DFlashTreeError::Backend)?;

        let mut tree_values = vec![0u32; weaver::TREE_LANE_COUNT * slots];
        for slot in 0..slots {
            tree_values[weaver::TREE_LANE_PARENT * slots + slot] = weaver::FRONTIER_NO_WINNER;
        }
        tree_values[weaver::TREE_LANE_TOKEN * slots] = params.bonus_token;
        tree_values[weaver::TREE_LANE_MASK * slots] = 1;

        let mut tree = encoder.allocate_constant_from_slice(&tree_values).map_err(DFlashTreeError::Backend)?;
        let mut frontier = encoder
            .allocate_constant_from_slice(&vec![0u32; weaver::FRONTIER_LANE_COUNT * capacity])
            .map_err(DFlashTreeError::Backend)?;
        let mut slot_ancestors =
            encoder.allocate_constant_from_slice(&vec![0u32; slots * stride]).map_err(DFlashTreeError::Backend)?;

        let mut round_token_id_values = vec![0u32; width];
        round_token_id_values[0] = params.bonus_token;
        let mut round_valid_values = vec![0u32; width];
        round_valid_values[0] = 1;
        let mut round_token_ids =
            encoder.allocate_constant_from_slice(&round_token_id_values).map_err(DFlashTreeError::Backend)?;
        let mut round_metadata = encoder
            .allocate_constant_from_slice(&vec![0u32; weaver::METADATA_LANE_COUNT * width])
            .map_err(DFlashTreeError::Backend)?;
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
                    &mut round_token_ids,
                    &mut round_metadata,
                    &mut round_ancestors,
                    &mut round_valid,
                    params.pool_ids,
                    params.pool_scores,
                    &mut round_candidate_ids,
                    &mut round_candidate_scores,
                    capacity as u32,
                    slots as u32,
                    rows as u32,
                    slot_start as u32,
                    stride as u32,
                    params.max_depth as u32,
                    params.lookahead_count as u32,
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
            let input = WeaverStepBatch {
                row_count: rows,
                candidate_count: pool_size,
                ancestor_stride: stride,
                parent_token_ids: &round_token_ids,
                candidate_ids,
                candidate_scores,
                ancestor_indices: &round_ancestors,
                node_metadata: &round_metadata,
            };
            let (child_ids, child_logprobs) = params.weaver.encode_step_batch(
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

fn tree_from_slots(tree: &[u32]) -> Vec<HostTreeNode> {
    assert!(
        tree.len().is_multiple_of(weaver::TREE_LANE_COUNT),
        "tree array must contain {} equal-length lanes",
        weaver::TREE_LANE_COUNT
    );
    let slots = tree.len() / weaver::TREE_LANE_COUNT;
    let lane = |lane: usize, slot: usize| tree[lane * slots + slot];
    let mut slot_to_node = vec![usize::MAX; slots];
    let mut nodes: Vec<HostTreeNode> = Vec::with_capacity(slots);
    for slot in 0..slots {
        if lane(weaver::TREE_LANE_MASK, slot) == 0 {
            continue;
        }
        let parent_slot = lane(weaver::TREE_LANE_PARENT, slot) as i32;
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
            token: lane(weaver::TREE_LANE_TOKEN, slot),
            parent,
            depth: lane(weaver::TREE_LANE_DEPTH, slot) as usize,
            logprob: f32::from_bits(lane(weaver::TREE_LANE_LOGPROB, slot)),
            children: Vec::new(),
        });
    }
    nodes
}

struct TreeEncodingArguments<'a, B: Backend> {
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
        self.speculator.propose_tree(state, bonus_token, self.target_embedding, options)
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
