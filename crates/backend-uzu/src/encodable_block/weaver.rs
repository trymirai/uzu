use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AllocationType, AsBufferRangeMut, Backend, Context, Encoder, Kernels,
        gpu_types::weaver::{
            CANDIDATES_MAX, FRONTIER_MAX_SLOTS, FRONTIER_NO_WINNER, FrontierIdx, MetadataIdx, TreeIdx,
        },
        kernel::{
            ActivationKernel, AncestorAttentionKernel, AttentionPrepareKernel, TensorAddBiasKernel,
            WeaverFrontierScatterKernel, WeaverFrontierSelectKernel, WeaverNodeCacheWriteKernel,
            WeaverTopChildrenKernel,
        },
    },
    config::{normalization::NormalizationConfig, weaver::WeaverConfig},
    data_type::DataType,
    encodable_block::{
        embedding::{Embedding, EmbeddingError},
        linear::{Linear, LinearBlockError},
        mixer::attention::{
            AttentionStateType,
            core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments, AttentionCores},
        },
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

const DATA_TYPE: DataType = DataType::BF16;

struct TopKChildren<B: Backend> {
    token_ids: Allocation<B>,
    logprobs: Allocation<B>,
}

struct PrefixOutput<B: Backend> {
    hidden: Allocation<B>,
    kv_cache: Allocation<B>,
}

struct PrefixAttentionInputs<B: Backend> {
    queries: Allocation<B>,
    kv_cache: Allocation<B>,
}

struct WeaverPrefixKvCache<B: Backend> {
    layer_kv: Box<[Allocation<B>]>,
    token_count: usize,
}

struct WeaverNodeKvCache<B: Backend> {
    layer_kv: Box<[Allocation<B>]>,
    node_capacity: u32,
}

struct WeaverStepBatch<'a, B: Backend> {
    node_count: usize,
    candidates_per_node: usize,
    ancestor_stride: usize,
    node_token_ids: &'a Allocation<B>,
    candidate_ids: &'a Allocation<B>,
    candidate_logits: &'a Allocation<B>,
    ancestor_indices: &'a Allocation<B>,
    node_metadata: &'a Allocation<B>,
    depth_seeds: &'a Allocation<B>,
}

pub(crate) struct WeaverTreeInput<'a, B: Backend> {
    pub(crate) target_hidden: &'a Allocation<B>,
    pub(crate) draft_hidden: &'a Allocation<B>,
    pub(crate) target_embedding: &'a Embedding<B>,
    pub(crate) candidate_ids: &'a Allocation<B>,
    pub(crate) candidate_scores: &'a Allocation<B>,
    pub(crate) candidate_rows: usize,
    pub(crate) candidates_per_row: usize,
    pub(crate) depth_seeds: &'a [u64],
    pub(crate) root_token_id: u32,
    pub(crate) tree_budget: usize,
    pub(crate) frontier_width: usize,
    pub(crate) children_per_node: usize,
}

pub(crate) struct EncodedWeaverTree<B: Backend> {
    slots: Allocation<B>,
}

pub(crate) struct ProposalNode {
    pub(crate) token: u32,
    pub(crate) depth: usize,
    pub(crate) children: Vec<usize>,
}

impl<B: Backend> EncodedWeaverTree<B> {
    pub(crate) fn decode(self) -> Vec<ProposalNode> {
        decode_tree_slots(&self.slots.copyout::<u32>())
    }
}

pub(crate) struct Weaver<B: Backend> {
    token_embedding_norm: Normalization<B>,
    token_embedding_projection: Box<dyn Linear<B>>,
    hidden_state_norm: Normalization<B>,
    hidden_state_projection: Box<dyn Linear<B>>,
    blocks: Box<[WeaverBlock<B>]>,
    readout_norm: Normalization<B>,
    readout_query_projection: Box<dyn Linear<B>>,
    position_embeddings: Allocation<B>,
    prefix_position_add: <B::Kernels as Kernels>::TensorAddBiasKernel,
    node_position_add: <B::Kernels as Kernels>::TensorAddBiasKernel,
    top_children: <B::Kernels as Kernels>::WeaverTopChildrenKernel,
    frontier_select: <B::Kernels as Kernels>::WeaverFrontierSelectKernel,
    frontier_scatter: <B::Kernels as Kernels>::WeaverFrontierScatterKernel,
    model_dim: usize,
    target_model_dim: usize,
    max_depth: usize,
}

struct WeaverBlock<B: Backend> {
    // Attention
    pre_attention_norm: Normalization<B>,
    qkv_projection: Box<dyn Linear<B>>,
    attention_prepare: <B::Kernels as Kernels>::AttentionPrepareKernel,
    prefix_attention: AttentionCores<B>,
    ancestor_attention: <B::Kernels as Kernels>::AncestorAttentionKernel,
    node_cache_write: <B::Kernels as Kernels>::WeaverNodeCacheWriteKernel,
    out_projection: Box<dyn Linear<B>>,

    // MLP
    pre_mlp_norm: Normalization<B>,
    up_projection: Box<dyn Linear<B>>,
    activation: <B::Kernels as Kernels>::ActivationKernel,
    down_projection: Box<dyn Linear<B>>,

    // Geometry
    attention_scale: f32,
    model_dim: usize,
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
}

#[derive(Debug, Error)]
pub enum WeaverNewError<B: Backend> {
    #[error("parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Weaver requires at least one layer")]
    InvalidLayerCount,
    #[error("model_dim must be divisible by num_heads")]
    InvalidHeadConfig,
    #[error("candidate_pool_size must be in 1..={max}, got {0}", max = CANDIDATES_MAX)]
    InvalidCandidatePoolSize(usize),
}

#[derive(Debug, Error)]
pub enum WeaverEncodeError<B: Backend> {
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
    #[error("invalid Weaver tree input")]
    InvalidTreeInput,
}

impl<B: Backend> Weaver<B> {
    pub(crate) fn new(
        context: &B::Context,
        config: &WeaverConfig,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, WeaverNewError<B>> {
        if config.num_layers == 0 {
            return Err(WeaverNewError::InvalidLayerCount);
        }
        if config.num_heads == 0 || !config.model_dim.is_multiple_of(config.num_heads) {
            return Err(WeaverNewError::InvalidHeadConfig);
        }
        if config.candidate_pool_size == 0 || config.candidate_pool_size > CANDIDATES_MAX {
            return Err(WeaverNewError::InvalidCandidatePoolSize(config.candidate_pool_size));
        }
        let token_embedding_norm = Normalization::new(
            config.target_embedding_dim,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            DATA_TYPE,
            &config.norm_config,
            &parameter_tree.subtree("embedding_norm")?,
            context,
        )?;
        let hidden_state_norm = Normalization::new(
            config.target_model_dim,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            DATA_TYPE,
            &config.norm_config,
            &parameter_tree.subtree("hidden_state_norm")?,
            context,
        )?;
        let token_embedding_projection = <dyn Linear<B>>::new(
            config.target_embedding_dim,
            [config.model_dim],
            true,
            context,
            DATA_TYPE,
            &parameter_tree.subtree("embedding_projection")?,
        )?;
        let blocks_tree = parameter_tree.subtree("blocks")?;
        let blocks = (0..config.num_layers)
            .map(|index| {
                WeaverBlock::new(
                    context,
                    config.model_dim,
                    config.hidden_dim,
                    config.num_heads,
                    index > 0,
                    &config.norm_config,
                    &blocks_tree.subtree(&index.to_string())?,
                )
            })
            .collect::<Result<Box<[_]>, WeaverNewError<B>>>()?;
        let readout_norm = Normalization::new(
            config.model_dim,
            None,
            ShortcutMode::Add,
            PostLayerScalar::None,
            DATA_TYPE,
            &config.norm_config,
            &parameter_tree.subtree("output_norm")?,
            context,
        )?;
        let hidden_state_projection = <dyn Linear<B>>::new(
            config.target_model_dim,
            [config.model_dim],
            true,
            context,
            DATA_TYPE,
            &parameter_tree.subtree("hidden_state_projection")?,
        )?;
        let readout_query_projection = <dyn Linear<B>>::new(
            config.model_dim,
            [config.target_model_dim],
            false,
            context,
            DATA_TYPE,
            &parameter_tree.subtree("query_projection")?,
        )?;
        let position_embeddings = parameter_tree
            .leaf("position_embeddings")?
            .validate(&[config.max_depth, config.model_dim], DataType::F32)?
            .read_allocation()?;
        let prefix_position_add =
            <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, DATA_TYPE, DataType::F32, true, false)
                .map_err(WeaverNewError::Backend)?;
        let node_position_add =
            <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, DATA_TYPE, DataType::F32, true, true)
                .map_err(WeaverNewError::Backend)?;
        let top_children =
            <B::Kernels as Kernels>::WeaverTopChildrenKernel::new(context).map_err(WeaverNewError::Backend)?;
        let frontier_select =
            <B::Kernels as Kernels>::WeaverFrontierSelectKernel::new(context).map_err(WeaverNewError::Backend)?;
        let frontier_scatter =
            <B::Kernels as Kernels>::WeaverFrontierScatterKernel::new(context).map_err(WeaverNewError::Backend)?;
        Ok(Self {
            token_embedding_norm,
            token_embedding_projection,
            hidden_state_norm,
            hidden_state_projection,
            blocks,
            readout_norm,
            readout_query_projection,
            position_embeddings,
            prefix_position_add,
            node_position_add,
            top_children,
            frontier_select,
            frontier_scatter,
            model_dim: config.model_dim,
            target_model_dim: config.target_model_dim,
            max_depth: config.max_depth,
        })
    }

    pub(crate) fn encode_tree(
        &self,
        input: WeaverTreeInput<'_, B>,
        context: &B::Context,
        encoder: &mut Encoder<B>,
    ) -> Result<EncodedWeaverTree<B>, WeaverEncodeError<B>> {
        let slot_count = input.tree_budget + 1;
        let frontier_capacity = slot_count * input.children_per_node;
        let lookahead_count = self.max_depth.min(input.candidate_rows);
        if input.tree_budget == 0
            || input.frontier_width == 0
            || input.children_per_node == 0
            || input.children_per_node > input.candidates_per_row
            || frontier_capacity > FRONTIER_MAX_SLOTS
            || lookahead_count == 0
            || input.depth_seeds.len() != self.max_depth
        {
            return Err(WeaverEncodeError::InvalidTreeInput);
        }

        let prefix = self.build_prefix(input.target_hidden, input.draft_hidden, lookahead_count, encoder)?;
        let mut node_cache = self.create_node_kv_cache(slot_count, context)?;

        let mut tree_values = vec![0u32; TreeIdx::COUNT * slot_count];
        for slot in 0..slot_count {
            tree_values[TreeIdx::ParentSlot as usize * slot_count + slot] = FRONTIER_NO_WINNER;
        }
        tree_values[TreeIdx::TokenId as usize * slot_count] = input.root_token_id;
        tree_values[TreeIdx::Valid as usize * slot_count] = 1;

        let mut tree = encoder.allocate_constant_from_slice(&tree_values).map_err(WeaverEncodeError::Backend)?;
        let mut frontier = encoder
            .allocate_constant_from_slice(&vec![0u32; FrontierIdx::COUNT * frontier_capacity])
            .map_err(WeaverEncodeError::Backend)?;
        let mut slot_ancestors = encoder
            .allocate_constant_from_slice(&vec![0u32; slot_count * self.max_depth])
            .map_err(WeaverEncodeError::Backend)?;

        let mut round_token_id_values = vec![0u32; input.frontier_width];
        round_token_id_values[0] = input.root_token_id;
        let mut round_valid_values = vec![0u32; input.frontier_width];
        round_valid_values[0] = 1;
        let mut round_token_ids =
            encoder.allocate_constant_from_slice(&round_token_id_values).map_err(WeaverEncodeError::Backend)?;
        let mut round_metadata = encoder
            .allocate_constant_from_slice(&vec![0u32; MetadataIdx::COUNT * input.frontier_width])
            .map_err(WeaverEncodeError::Backend)?;
        let mut round_ancestors = encoder
            .allocate_constant_from_slice(&vec![0u32; input.frontier_width * self.max_depth])
            .map_err(WeaverEncodeError::Backend)?;
        let mut round_valid =
            encoder.allocate_constant_from_slice(&round_valid_values).map_err(WeaverEncodeError::Backend)?;
        let mut round_candidate_ids = encoder
            .allocate_constant_from_slice(&vec![0u32; input.frontier_width * input.candidates_per_row])
            .map_err(WeaverEncodeError::Backend)?;
        let mut round_candidate_scores = encoder
            .allocate_constant_from_slice(&vec![0.0f32; input.frontier_width * input.candidates_per_row])
            .map_err(WeaverEncodeError::Backend)?;
        let depth_seeds =
            encoder.allocate_constant_from_slice(input.depth_seeds).map_err(WeaverEncodeError::Backend)?;

        let mut slot_start = 0;
        while slot_start < slot_count {
            let node_count = if slot_start == 0 {
                1
            } else {
                input.frontier_width.min(slot_count - slot_start)
            };
            if slot_start > 0 {
                self.frontier_select.encode(
                    &mut frontier,
                    &mut tree,
                    &mut slot_ancestors,
                    &mut round_token_ids,
                    &mut round_metadata,
                    &mut round_ancestors,
                    &mut round_valid,
                    input.candidate_ids,
                    input.candidate_scores,
                    &mut round_candidate_ids,
                    &mut round_candidate_scores,
                    frontier_capacity as u32,
                    slot_count as u32,
                    node_count as u32,
                    slot_start as u32,
                    self.max_depth as u32,
                    self.max_depth as u32,
                    lookahead_count as u32,
                    input.candidate_rows as u32,
                    input.candidates_per_row as u32,
                    encoder,
                );
            }
            let (candidate_ids, candidate_scores) = if slot_start == 0 {
                (input.candidate_ids, input.candidate_scores)
            } else {
                (&round_candidate_ids, &round_candidate_scores)
            };
            let batch = WeaverStepBatch {
                node_count,
                candidates_per_node: input.candidates_per_row,
                ancestor_stride: self.max_depth,
                node_token_ids: &round_token_ids,
                candidate_ids,
                candidate_logits: candidate_scores,
                ancestor_indices: &round_ancestors,
                node_metadata: &round_metadata,
                depth_seeds: &depth_seeds,
            };
            let children = self.encode_step_batch(
                &prefix,
                &batch,
                &mut node_cache,
                input.children_per_node,
                input.target_embedding,
                encoder,
            )?;
            self.frontier_scatter.encode(
                &tree,
                &round_metadata,
                &round_valid,
                &children.token_ids,
                &children.logprobs,
                &mut frontier,
                frontier_capacity as u32,
                slot_count as u32,
                node_count as u32,
                input.children_per_node as u32,
                encoder,
            );
            slot_start += node_count;
        }
        Ok(EncodedWeaverTree {
            slots: tree,
        })
    }

    fn build_prefix(
        &self,
        target_hidden: &Allocation<B>,
        draft_hidden: &Allocation<B>,
        lookahead_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<WeaverPrefixKvCache<B>, WeaverEncodeError<B>> {
        assert!((1..=self.max_depth).contains(&lookahead_count));
        let token_count = lookahead_count + 1;
        let hidden_row_bytes = self.target_model_dim * DATA_TYPE.size_in_bytes();

        let mut prefix_input = encoder
            .allocate_scratch(size_for_shape(&[token_count, self.target_model_dim], DATA_TYPE))
            .map_err(WeaverEncodeError::Backend)?;
        encoder.encode_copy(target_hidden, 0..hidden_row_bytes, &mut prefix_input, 0..hidden_row_bytes);
        encoder.encode_copy(
            draft_hidden,
            hidden_row_bytes..token_count * hidden_row_bytes,
            &mut prefix_input,
            hidden_row_bytes..token_count * hidden_row_bytes,
        );
        let normalized_prefix = self
            .hidden_state_norm
            .encode(&prefix_input, 0, token_count, None, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let mut hidden = self
            .hidden_state_projection
            .encode(normalized_prefix, token_count, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let position_elements = lookahead_count * self.model_dim;
        self.prefix_position_add.encode(
            None::<&Allocation<B>>,
            &self.position_embeddings,
            None::<&Allocation<B>>,
            (&mut hidden, self.model_dim * DATA_TYPE.size_in_bytes()),
            position_elements as u32,
            position_elements as u32,
            encoder,
        );
        let (last_block, preceding_blocks) = self.blocks.split_last().expect("Weaver must have at least one block");
        let mut residual = encoder.allocate_scratch(hidden.size()).map_err(WeaverEncodeError::Backend)?;
        let mut layer_kv = Vec::with_capacity(self.blocks.len());
        for block in preceding_blocks {
            let PrefixOutput {
                hidden: next_hidden,
                kv_cache,
            } = block.encode_prefix(hidden, &mut residual, token_count, encoder).map_err(WeaverEncodeError::Backend)?;
            layer_kv.push(kv_cache);
            hidden = next_hidden;
        }
        layer_kv.push(
            last_block
                .prepare_prefix_attention(&hidden, &mut residual, token_count, encoder)
                .map_err(WeaverEncodeError::Backend)?
                .kv_cache,
        );
        Ok(WeaverPrefixKvCache {
            layer_kv: layer_kv.into_boxed_slice(),
            token_count,
        })
    }

    fn create_node_kv_cache(
        &self,
        capacity: usize,
        context: &B::Context,
    ) -> Result<WeaverNodeKvCache<B>, WeaverEncodeError<B>> {
        assert!(capacity > 0, "Weaver node capacity must be positive");
        let node_capacity = u32::try_from(capacity).expect("Weaver node capacity exceeds the kernel limit");
        let kv_size = size_for_shape(&[2, capacity, self.model_dim], DATA_TYPE);
        let layer_kv = (0..self.blocks.len())
            .map(|_| context.create_allocation(kv_size, AllocationType::Global).map_err(WeaverEncodeError::Backend))
            .collect::<Result<Box<[_]>, _>>()?;
        Ok(WeaverNodeKvCache {
            layer_kv,
            node_capacity,
        })
    }

    fn encode_step_batch(
        &self,
        prefix: &WeaverPrefixKvCache<B>,
        batch: &WeaverStepBatch<'_, B>,
        node_cache: &mut WeaverNodeKvCache<B>,
        children_per_node: usize,
        target_embedding: &Embedding<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<TopKChildren<B>, WeaverEncodeError<B>> {
        let token_embedding = target_embedding.encode_lookup(batch.node_token_ids, batch.node_count, encoder)?;
        let normalized_embedding = self
            .token_embedding_norm
            .encode(&token_embedding, 0, batch.node_count, None, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let mut hidden = self
            .token_embedding_projection
            .encode(normalized_embedding, batch.node_count, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        self.node_position_add.encode(
            None::<&Allocation<B>>,
            &self.position_embeddings,
            Some(batch.node_metadata),
            &mut hidden,
            self.model_dim as u32,
            (batch.node_count * self.model_dim) as u32,
            encoder,
        );

        let mut residual = encoder.allocate_scratch(hidden.size()).map_err(WeaverEncodeError::Backend)?;
        let node_capacity = node_cache.node_capacity;
        for (layer_index, block) in self.blocks.iter().enumerate() {
            hidden = block
                .encode_step(
                    hidden,
                    &mut residual,
                    &prefix.layer_kv[layer_index],
                    &mut node_cache.layer_kv[layer_index],
                    node_capacity,
                    batch,
                    prefix.token_count,
                    encoder,
                )
                .map_err(WeaverEncodeError::Backend)?;
        }

        self.encode_step_output(&hidden, &mut residual, batch, children_per_node, target_embedding, encoder)
    }

    /// Outputs must outlive the batch.
    fn encode_step_output(
        &self,
        hidden: &Allocation<B>,
        residual: &mut Allocation<B>,
        batch: &WeaverStepBatch<'_, B>,
        children_per_node: usize,
        target_embedding: &Embedding<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<TopKChildren<B>, WeaverEncodeError<B>> {
        let normalized_output = self
            .readout_norm
            .encode(hidden, 0, batch.node_count, Some(residual), encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let query = self
            .readout_query_projection
            .encode(normalized_output, batch.node_count, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let residual_logits = target_embedding.encode_readout_sparse(
            &query,
            batch.candidate_ids,
            batch.node_count,
            batch.candidates_per_node,
            encoder,
        )?;
        let mut token_ids = encoder
            .allocate_scratch(size_for_shape(&[batch.node_count, children_per_node], DataType::U32))
            .map_err(WeaverEncodeError::Backend)?;
        let mut logprobs = encoder
            .allocate_scratch(size_for_shape(&[batch.node_count, children_per_node], DataType::F32))
            .map_err(WeaverEncodeError::Backend)?;
        self.top_children.encode(
            &residual_logits,
            batch.candidate_logits,
            batch.candidate_ids,
            batch.depth_seeds,
            batch.node_metadata,
            &mut token_ids,
            &mut logprobs,
            batch.node_count as u32,
            batch.candidates_per_node as u32,
            children_per_node as u32,
            target_embedding.vocab_size() as u32,
            encoder,
        );
        Ok(TopKChildren {
            token_ids,
            logprobs,
        })
    }
}

impl<B: Backend> WeaverBlock<B> {
    fn new(
        context: &B::Context,
        model_dim: usize,
        hidden_dim: usize,
        num_heads: usize,
        add_to_residual: bool,
        norm_config: &NormalizationConfig,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, WeaverNewError<B>> {
        let head_dim = model_dim / num_heads;
        let attention_scale = 1.0 / (head_dim as f32).sqrt();
        let qkv_projection = <dyn Linear<B>>::new(
            model_dim,
            [3 * model_dim],
            false,
            context,
            DATA_TYPE,
            &parameter_tree.subtree("qkv_projection")?,
        )?;
        let out_projection = <dyn Linear<B>>::new(
            model_dim,
            [model_dim],
            false,
            context,
            DATA_TYPE,
            &parameter_tree.subtree("out_projection")?,
        )?;
        let prefix_attention = AttentionCores::new(
            AttentionCoreNewArguments {
                head_dim,
                num_groups: num_heads,
                num_q_heads: num_heads,
                has_sinks: false,
                is_kv_cache_ring: false,
                is_causal: true,
                is_trie: false,
                sliding_window_size: None,
                scale: Some(attention_scale),
                data_type: DATA_TYPE,
            },
            context,
        )
        .map_err(WeaverNewError::Backend)?;
        let attention_prepare =
            <B::Kernels as Kernels>::AttentionPrepareKernel::new(context, DATA_TYPE, DataType::F32, true, false)
                .map_err(WeaverNewError::Backend)?;
        let pre_attention_norm = Normalization::new(
            model_dim,
            None,
            if add_to_residual {
                ShortcutMode::Add
            } else {
                ShortcutMode::Copy
            },
            PostLayerScalar::None,
            DATA_TYPE,
            norm_config,
            &parameter_tree.subtree("pre_attention_norm")?,
            context,
        )?;
        let pre_mlp_norm = Normalization::new(
            model_dim,
            None,
            ShortcutMode::Add,
            PostLayerScalar::None,
            DATA_TYPE,
            norm_config,
            &parameter_tree.subtree("pre_mlp_norm")?,
            context,
        )?;
        let up_projection = <dyn Linear<B>>::new(
            model_dim,
            [hidden_dim],
            true,
            context,
            DATA_TYPE,
            &parameter_tree.subtree("up_projection")?,
        )?;
        let down_projection = <dyn Linear<B>>::new(
            hidden_dim,
            [model_dim],
            true,
            context,
            DATA_TYPE,
            &parameter_tree.subtree("down_projection")?,
        )?;
        let activation = <B::Kernels as Kernels>::ActivationKernel::new(context, DATA_TYPE, true)
            .map_err(WeaverNewError::Backend)?;
        let ancestor_attention =
            <B::Kernels as Kernels>::AncestorAttentionKernel::new(context, head_dim as u32, num_heads as u32)
                .map_err(WeaverNewError::Backend)?;
        let node_cache_write = <B::Kernels as Kernels>::WeaverNodeCacheWriteKernel::new(context, DATA_TYPE)
            .map_err(WeaverNewError::Backend)?;
        Ok(Self {
            qkv_projection,
            out_projection,
            prefix_attention,
            attention_prepare,
            pre_attention_norm,
            pre_mlp_norm,
            up_projection,
            down_projection,
            activation,
            ancestor_attention,
            node_cache_write,
            attention_scale,
            model_dim,
            hidden_dim,
            num_heads,
            head_dim,
        })
    }

    fn encode_prefix(
        &self,
        input: Allocation<B>,
        residual: &mut Allocation<B>,
        token_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<PrefixOutput<B>, B::Error> {
        let PrefixAttentionInputs {
            queries,
            kv_cache,
        } = self.prepare_prefix_attention(&input, residual, token_count, encoder)?;
        let state_type = AttentionStateType::Full {
            length: 0,
        };
        let kv_plane_bytes = size_for_shape(&[token_count, self.model_dim], DATA_TYPE);
        let attention = self.prefix_attention.encode(
            AttentionCoreEncodeArguments {
                queries: &queries,
                keys: &kv_cache,
                values: (&kv_cache, kv_plane_bytes),
                suffix_length: token_count,
                trie: None,
                sinks: None,
                state_type: &state_type,
            },
            encoder,
        )?;
        let hidden = self.encode_post_attention(attention, residual, token_count, encoder)?;
        Ok(PrefixOutput {
            hidden,
            kv_cache,
        })
    }

    fn prepare_prefix_attention(
        &self,
        input: &Allocation<B>,
        residual: &mut Allocation<B>,
        token_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<PrefixAttentionInputs<B>, B::Error> {
        let attention_input = self.pre_attention_norm.encode(input, 0, token_count, Some(residual), encoder)?;
        let qkv = self.qkv_projection.encode(attention_input, token_count, encoder)?;
        let mut queries =
            encoder.allocate_scratch(size_for_shape(&[self.num_heads, token_count, self.head_dim], DATA_TYPE))?;
        let kv_plane_bytes = size_for_shape(&[token_count, self.model_dim], DATA_TYPE);
        let mut kv = encoder
            .context()
            .create_allocation(size_for_shape(&[2, token_count, self.model_dim], DATA_TYPE), AllocationType::Global)?;
        let (keys, values) = kv.as_buffer_range_mut().split_at(kv_plane_bytes);
        self.attention_prepare.encode(
            &qkv,
            &mut queries,
            Some(keys),
            Some(values),
            None::<&Allocation<B>>,
            None::<&Allocation<B>>,
            self.num_heads as u32,
            Some(self.num_heads as u32),
            self.head_dim as u32,
            None,
            Some(0),
            token_count as u32,
            encoder,
        );
        Ok(PrefixAttentionInputs {
            queries,
            kv_cache: kv,
        })
    }

    fn encode_step(
        &self,
        input: Allocation<B>,
        residual: &mut Allocation<B>,
        prefix_kv: &Allocation<B>,
        node_kv_cache: &mut Allocation<B>,
        node_capacity: u32,
        batch: &WeaverStepBatch<'_, B>,
        prefix_length: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let attention_input = self.pre_attention_norm.encode(&input, 0, batch.node_count, Some(residual), encoder)?;
        let current_qkv = self.qkv_projection.encode(attention_input, batch.node_count, encoder)?;
        let metadata_lane_bytes = batch.node_count * DataType::U32.size_in_bytes();
        let ancestor_counts = (batch.node_metadata, MetadataIdx::AncestorCount as usize * metadata_lane_bytes);
        let tree_slots = (batch.node_metadata, MetadataIdx::TreeSlot as usize * metadata_lane_bytes);
        let mut attention = encoder.allocate_scratch(size_for_shape(&[batch.node_count, self.model_dim], DATA_TYPE))?;
        self.ancestor_attention.encode(
            prefix_kv,
            &*node_kv_cache,
            &current_qkv,
            batch.ancestor_indices,
            ancestor_counts,
            &mut attention,
            batch.node_count as u32,
            prefix_length as u32,
            batch.ancestor_stride as u32,
            node_capacity,
            self.attention_scale,
            encoder,
        );
        // Separate dispatch so the node arena is read-only above; the encoder
        // serializes it after the attention that reads the ancestor slots.
        self.node_cache_write.encode(
            &current_qkv,
            node_kv_cache,
            tree_slots,
            self.model_dim as u32,
            node_capacity,
            (batch.node_count * 2 * self.model_dim) as u32,
            encoder,
        );
        self.encode_post_attention(attention, residual, batch.node_count, encoder)
    }

    fn encode_post_attention(
        &self,
        attention: Allocation<B>,
        residual: &mut Allocation<B>,
        row_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let projected_attention = self.out_projection.encode(attention, row_count, encoder)?;
        let mlp_input = self.pre_mlp_norm.encode(&projected_attention, 0, row_count, Some(residual), encoder)?;
        let mut mlp_hidden = self.up_projection.encode(mlp_input, row_count, encoder)?;
        self.activation.encode(
            None::<&Allocation<B>>,
            &mut mlp_hidden,
            (row_count * self.hidden_dim) as u32,
            crate::backends::common::gpu_types::ActivationType::GELUExact,
            encoder,
        );
        self.down_projection.encode(mlp_hidden, row_count, encoder)
    }
}

fn decode_tree_slots(tree: &[u32]) -> Vec<ProposalNode> {
    assert!(tree.len().is_multiple_of(TreeIdx::COUNT), "tree array must contain {} equal-length lanes", TreeIdx::COUNT);
    let slot_count = tree.len() / TreeIdx::COUNT;
    let field = |field: TreeIdx, slot: usize| tree[field as usize * slot_count + slot];
    let mut slot_to_node = vec![usize::MAX; slot_count];
    let mut nodes: Vec<ProposalNode> = Vec::with_capacity(slot_count);
    for slot in 0..slot_count {
        if field(TreeIdx::Valid, slot) == 0 {
            continue;
        }
        let parent_slot = field(TreeIdx::ParentSlot, slot) as i32;
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
        nodes.push(ProposalNode {
            token: field(TreeIdx::TokenId, slot),
            depth: field(TreeIdx::Depth, slot) as usize,
            children: Vec::new(),
        });
    }
    nodes
}
