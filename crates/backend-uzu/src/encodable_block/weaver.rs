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
    model_logprobs: Allocation<B>,
}

struct PrefixLayerOutput<B: Backend> {
    mlp_delta: Allocation<B>,
    kv_cache: Allocation<B>,
}

struct PreparedPrefixAttention<B: Backend> {
    queries: Allocation<B>,
    kv_cache: Allocation<B>,
}

struct PrefixKvCache<B: Backend> {
    layers: Box<[Allocation<B>]>,
    length: usize,
}

struct NodeKvCache<B: Backend> {
    layers: Box<[Allocation<B>]>,
    capacity: u32,
}

struct NodeBatch<'a, B: Backend> {
    count: usize,
    token_ids: &'a Allocation<B>,
    metadata: &'a Allocation<B>,
    ancestor_indices: &'a Allocation<B>,
    ancestor_stride: usize,
}

struct CandidateBatch<'a, B: Backend> {
    ids: &'a Allocation<B>,
    logits: &'a Allocation<B>,
    candidates_per_node: usize,
    depth_seeds: &'a Allocation<B>,
}

pub struct CandidatePool<'a, B: Backend> {
    ids: &'a Allocation<B>,
    logits: &'a Allocation<B>,
    depth_count: usize,
    candidates_per_depth: usize,
}

impl<'a, B: Backend> CandidatePool<'a, B> {
    pub fn new(
        ids: &'a Allocation<B>,
        logits: &'a Allocation<B>,
        depth_count: usize,
        candidates_per_depth: usize,
    ) -> Self {
        Self {
            ids,
            logits,
            depth_count,
            candidates_per_depth,
        }
    }
}

pub struct WeaverInputs<'a, B: Backend> {
    target_hidden: &'a Allocation<B>,
    draft_hidden: &'a Allocation<B>,
    target_embedding: &'a Embedding<B>,
    candidates: CandidatePool<'a, B>,
    depth_seeds: &'a [u64],
    root_token_id: u32,
}

impl<'a, B: Backend> WeaverInputs<'a, B> {
    pub fn new(
        target_hidden: &'a Allocation<B>,
        draft_hidden: &'a Allocation<B>,
        target_embedding: &'a Embedding<B>,
        candidates: CandidatePool<'a, B>,
        depth_seeds: &'a [u64],
        root_token_id: u32,
    ) -> Self {
        Self {
            target_hidden,
            draft_hidden,
            target_embedding,
            candidates,
            depth_seeds,
            root_token_id,
        }
    }
}

pub struct TreeShape {
    budget: usize,
    frontier_width: usize,
    children_per_node: usize,
}

impl TreeShape {
    pub fn new(
        budget: usize,
        frontier_width: usize,
        children_per_node: usize,
    ) -> Self {
        Self {
            budget,
            frontier_width,
            children_per_node,
        }
    }
}

pub struct EncodedWeaverTree<B: Backend> {
    packed_tree: Allocation<B>,
}

pub struct ProposalNode {
    pub token_id: u32,
    pub depth: usize,
    pub child_indices: Vec<usize>,
}

impl<B: Backend> EncodedWeaverTree<B> {
    pub fn read_nodes(self) -> Vec<ProposalNode> {
        nodes_from_packed_tree(&self.packed_tree.copyout::<u32>())
    }
}

pub struct Weaver<B: Backend> {
    token_embedding_norm: Normalization<B>,
    token_embedding_projection: Box<dyn Linear<B>>,
    hidden_state_norm: Normalization<B>,
    hidden_state_projection: Box<dyn Linear<B>>,
    layers: Box<[WeaverLayer<B>]>,
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

struct WeaverLayer<B: Backend> {
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
    pub fn new(
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
        let layer_parameters = parameter_tree.subtree("blocks")?;
        let layers = (0..config.num_layers)
            .map(|index| {
                WeaverLayer::new(
                    context,
                    config.model_dim,
                    config.hidden_dim,
                    config.num_heads,
                    index > 0,
                    &config.norm_config,
                    &layer_parameters.subtree(&index.to_string())?,
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
            layers,
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

    pub fn encode_tree(
        &self,
        inputs: WeaverInputs<'_, B>,
        shape: TreeShape,
        context: &B::Context,
        encoder: &mut Encoder<B>,
    ) -> Result<EncodedWeaverTree<B>, WeaverEncodeError<B>> {
        let tree_slot_count = shape.budget + 1;
        let frontier_capacity = tree_slot_count * shape.children_per_node;
        let lookahead_count = self.max_depth.min(inputs.candidates.depth_count);
        let ancestor_stride = self.max_depth;
        if shape.budget == 0
            || shape.frontier_width == 0
            || shape.children_per_node == 0
            || shape.children_per_node > inputs.candidates.candidates_per_depth
            || frontier_capacity > FRONTIER_MAX_SLOTS
            || lookahead_count == 0
            || inputs.depth_seeds.len() != self.max_depth
        {
            return Err(WeaverEncodeError::InvalidTreeInput);
        }

        let prefix_cache = self.build_prefix(inputs.target_hidden, inputs.draft_hidden, lookahead_count, encoder)?;
        let mut node_cache = self.create_node_kv_cache(tree_slot_count, context)?;

        let mut tree_init = vec![0u32; TreeIdx::COUNT * tree_slot_count];
        for slot in 0..tree_slot_count {
            tree_init[TreeIdx::ParentSlot as usize * tree_slot_count + slot] = FRONTIER_NO_WINNER;
        }
        tree_init[TreeIdx::TokenId as usize * tree_slot_count] = inputs.root_token_id;
        tree_init[TreeIdx::Valid as usize * tree_slot_count] = 1;

        let mut packed_tree = encoder.allocate_constant_from_slice(&tree_init).map_err(WeaverEncodeError::Backend)?;
        let mut frontier = encoder
            .allocate_constant_from_slice(&vec![0u32; FrontierIdx::COUNT * frontier_capacity])
            .map_err(WeaverEncodeError::Backend)?;
        let mut slot_ancestors = encoder
            .allocate_constant_from_slice(&vec![0u32; tree_slot_count * ancestor_stride])
            .map_err(WeaverEncodeError::Backend)?;

        let mut initial_node_token_ids = vec![0u32; shape.frontier_width];
        initial_node_token_ids[0] = inputs.root_token_id;
        let mut initial_node_valid = vec![0u32; shape.frontier_width];
        initial_node_valid[0] = 1;
        let mut node_token_ids =
            encoder.allocate_constant_from_slice(&initial_node_token_ids).map_err(WeaverEncodeError::Backend)?;
        let mut node_metadata = encoder
            .allocate_constant_from_slice(&vec![0u32; MetadataIdx::COUNT * shape.frontier_width])
            .map_err(WeaverEncodeError::Backend)?;
        let mut node_ancestor_indices = encoder
            .allocate_constant_from_slice(&vec![0u32; shape.frontier_width * ancestor_stride])
            .map_err(WeaverEncodeError::Backend)?;
        let mut node_valid =
            encoder.allocate_constant_from_slice(&initial_node_valid).map_err(WeaverEncodeError::Backend)?;
        let mut node_candidate_ids = encoder
            .allocate_constant_from_slice(&vec![0u32; shape.frontier_width * inputs.candidates.candidates_per_depth])
            .map_err(WeaverEncodeError::Backend)?;
        let mut node_candidate_logits = encoder
            .allocate_constant_from_slice(&vec![0.0f32; shape.frontier_width * inputs.candidates.candidates_per_depth])
            .map_err(WeaverEncodeError::Backend)?;
        let depth_seeds =
            encoder.allocate_constant_from_slice(inputs.depth_seeds).map_err(WeaverEncodeError::Backend)?;

        let mut batch_start_slot = 0;
        while batch_start_slot < tree_slot_count {
            let batch_node_count = if batch_start_slot == 0 {
                1
            } else {
                shape.frontier_width.min(tree_slot_count - batch_start_slot)
            };
            if batch_start_slot > 0 {
                self.frontier_select.encode(
                    &mut frontier,
                    &mut packed_tree,
                    &mut slot_ancestors,
                    &mut node_token_ids,
                    &mut node_metadata,
                    &mut node_ancestor_indices,
                    &mut node_valid,
                    inputs.candidates.ids,
                    inputs.candidates.logits,
                    &mut node_candidate_ids,
                    &mut node_candidate_logits,
                    frontier_capacity as u32,
                    tree_slot_count as u32,
                    batch_node_count as u32,
                    batch_start_slot as u32,
                    ancestor_stride as u32,
                    self.max_depth as u32,
                    lookahead_count as u32,
                    inputs.candidates.depth_count as u32,
                    inputs.candidates.candidates_per_depth as u32,
                    encoder,
                );
            }
            let (batch_candidate_ids, batch_candidate_logits) = if batch_start_slot == 0 {
                (inputs.candidates.ids, inputs.candidates.logits)
            } else {
                (&node_candidate_ids, &node_candidate_logits)
            };
            let nodes = NodeBatch {
                count: batch_node_count,
                token_ids: &node_token_ids,
                metadata: &node_metadata,
                ancestor_indices: &node_ancestor_indices,
                ancestor_stride,
            };
            let candidates = CandidateBatch {
                ids: batch_candidate_ids,
                logits: batch_candidate_logits,
                candidates_per_node: inputs.candidates.candidates_per_depth,
                depth_seeds: &depth_seeds,
            };
            let selected_children = self.expand_nodes(
                &prefix_cache,
                &nodes,
                &candidates,
                &mut node_cache,
                shape.children_per_node,
                inputs.target_embedding,
                encoder,
            )?;
            self.frontier_scatter.encode(
                &packed_tree,
                &node_metadata,
                &node_valid,
                &selected_children.token_ids,
                &selected_children.model_logprobs,
                &mut frontier,
                frontier_capacity as u32,
                tree_slot_count as u32,
                batch_node_count as u32,
                shape.children_per_node as u32,
                encoder,
            );
            batch_start_slot += batch_node_count;
        }
        Ok(EncodedWeaverTree {
            packed_tree,
        })
    }

    fn build_prefix(
        &self,
        target_hidden: &Allocation<B>,
        draft_hidden: &Allocation<B>,
        lookahead_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<PrefixKvCache<B>, WeaverEncodeError<B>> {
        assert!((1..=self.max_depth).contains(&lookahead_count));
        let token_count = lookahead_count + 1;
        let hidden_row_bytes = self.target_model_dim * DATA_TYPE.size_in_bytes();

        let mut prefix_hidden = encoder
            .allocate_scratch(size_for_shape(&[token_count, self.target_model_dim], DATA_TYPE))
            .map_err(WeaverEncodeError::Backend)?;
        encoder.encode_copy(target_hidden, 0..hidden_row_bytes, &mut prefix_hidden, 0..hidden_row_bytes);
        encoder.encode_copy(
            draft_hidden,
            hidden_row_bytes..token_count * hidden_row_bytes,
            &mut prefix_hidden,
            hidden_row_bytes..token_count * hidden_row_bytes,
        );
        let normalized_prefix = self
            .hidden_state_norm
            .encode(&prefix_hidden, 0, token_count, None, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let mut residual_input = self
            .hidden_state_projection
            .encode(normalized_prefix, token_count, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let position_elements = lookahead_count * self.model_dim;
        self.prefix_position_add.encode(
            None::<&Allocation<B>>,
            &self.position_embeddings,
            None::<&Allocation<B>>,
            (&mut residual_input, self.model_dim * DATA_TYPE.size_in_bytes()),
            position_elements as u32,
            position_elements as u32,
            encoder,
        );
        let (last_layer, preceding_layers) = self.layers.split_last().expect("Weaver must have at least one layer");
        let mut residual_state = encoder.allocate_scratch(residual_input.size()).map_err(WeaverEncodeError::Backend)?;
        let mut prefix_layers = Vec::with_capacity(self.layers.len());
        for layer in preceding_layers {
            let PrefixLayerOutput {
                mlp_delta,
                kv_cache,
            } = layer
                .encode_prefix(residual_input, &mut residual_state, token_count, encoder)
                .map_err(WeaverEncodeError::Backend)?;
            prefix_layers.push(kv_cache);
            residual_input = mlp_delta;
        }
        prefix_layers.push(
            last_layer
                .prepare_prefix_attention(&residual_input, &mut residual_state, token_count, encoder)
                .map_err(WeaverEncodeError::Backend)?
                .kv_cache,
        );
        Ok(PrefixKvCache {
            layers: prefix_layers.into_boxed_slice(),
            length: token_count,
        })
    }

    fn create_node_kv_cache(
        &self,
        capacity: usize,
        context: &B::Context,
    ) -> Result<NodeKvCache<B>, WeaverEncodeError<B>> {
        assert!(capacity > 0, "Weaver node capacity must be positive");
        let kernel_capacity = u32::try_from(capacity).expect("Weaver node capacity exceeds the kernel limit");
        let kv_size = size_for_shape(&[2, capacity, self.model_dim], DATA_TYPE);
        let layers = (0..self.layers.len())
            .map(|_| context.create_allocation(kv_size, AllocationType::Global).map_err(WeaverEncodeError::Backend))
            .collect::<Result<Box<[_]>, _>>()?;
        Ok(NodeKvCache {
            layers,
            capacity: kernel_capacity,
        })
    }

    fn expand_nodes(
        &self,
        prefix_cache: &PrefixKvCache<B>,
        nodes: &NodeBatch<'_, B>,
        candidates: &CandidateBatch<'_, B>,
        node_cache: &mut NodeKvCache<B>,
        children_per_node: usize,
        target_embedding: &Embedding<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<TopKChildren<B>, WeaverEncodeError<B>> {
        let token_embedding = target_embedding.encode_lookup(nodes.token_ids, nodes.count, encoder)?;
        let normalized_embedding = self
            .token_embedding_norm
            .encode(&token_embedding, 0, nodes.count, None, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let mut residual_input = self
            .token_embedding_projection
            .encode(normalized_embedding, nodes.count, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        self.node_position_add.encode(
            None::<&Allocation<B>>,
            &self.position_embeddings,
            Some(nodes.metadata),
            &mut residual_input,
            self.model_dim as u32,
            (nodes.count * self.model_dim) as u32,
            encoder,
        );

        let mut residual_state = encoder.allocate_scratch(residual_input.size()).map_err(WeaverEncodeError::Backend)?;
        let node_capacity = node_cache.capacity;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            residual_input = layer
                .encode_nodes(
                    residual_input,
                    &mut residual_state,
                    &prefix_cache.layers[layer_index],
                    &mut node_cache.layers[layer_index],
                    node_capacity,
                    nodes,
                    prefix_cache.length,
                    encoder,
                )
                .map_err(WeaverEncodeError::Backend)?;
        }

        self.select_children(
            &residual_input,
            &mut residual_state,
            nodes,
            candidates,
            children_per_node,
            target_embedding,
            encoder,
        )
    }

    fn select_children(
        &self,
        final_delta: &Allocation<B>,
        residual_state: &mut Allocation<B>,
        nodes: &NodeBatch<'_, B>,
        candidates: &CandidateBatch<'_, B>,
        children_per_node: usize,
        target_embedding: &Embedding<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<TopKChildren<B>, WeaverEncodeError<B>> {
        let normalized_output = self
            .readout_norm
            .encode(final_delta, 0, nodes.count, Some(residual_state), encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let query = self
            .readout_query_projection
            .encode(normalized_output, nodes.count, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let logit_residuals = target_embedding.encode_readout_sparse(
            &query,
            candidates.ids,
            nodes.count,
            candidates.candidates_per_node,
            encoder,
        )?;
        let mut token_ids = encoder
            .allocate_scratch(size_for_shape(&[nodes.count, children_per_node], DataType::U32))
            .map_err(WeaverEncodeError::Backend)?;
        let mut model_logprobs = encoder
            .allocate_scratch(size_for_shape(&[nodes.count, children_per_node], DataType::F32))
            .map_err(WeaverEncodeError::Backend)?;
        self.top_children.encode(
            &logit_residuals,
            candidates.logits,
            candidates.ids,
            candidates.depth_seeds,
            nodes.metadata,
            &mut token_ids,
            &mut model_logprobs,
            nodes.count as u32,
            candidates.candidates_per_node as u32,
            children_per_node as u32,
            target_embedding.vocab_size() as u32,
            encoder,
        );
        Ok(TopKChildren {
            token_ids,
            model_logprobs,
        })
    }
}

impl<B: Backend> WeaverLayer<B> {
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
        residual_input: Allocation<B>,
        residual_state: &mut Allocation<B>,
        token_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<PrefixLayerOutput<B>, B::Error> {
        let PreparedPrefixAttention {
            queries,
            kv_cache,
        } = self.prepare_prefix_attention(&residual_input, residual_state, token_count, encoder)?;
        let state_type = AttentionStateType::Full {
            length: 0,
        };
        let kv_plane_bytes = size_for_shape(&[token_count, self.model_dim], DATA_TYPE);
        let attention_output = self.prefix_attention.encode(
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
        let mlp_delta = self.encode_post_attention(attention_output, residual_state, token_count, encoder)?;
        Ok(PrefixLayerOutput {
            mlp_delta,
            kv_cache,
        })
    }

    fn prepare_prefix_attention(
        &self,
        residual_input: &Allocation<B>,
        residual_state: &mut Allocation<B>,
        token_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<PreparedPrefixAttention<B>, B::Error> {
        let attention_input =
            self.pre_attention_norm.encode(residual_input, 0, token_count, Some(residual_state), encoder)?;
        let qkv = self.qkv_projection.encode(attention_input, token_count, encoder)?;
        let mut queries =
            encoder.allocate_scratch(size_for_shape(&[self.num_heads, token_count, self.head_dim], DATA_TYPE))?;
        let kv_plane_bytes = size_for_shape(&[token_count, self.model_dim], DATA_TYPE);
        let mut kv_cache = encoder
            .context()
            .create_allocation(size_for_shape(&[2, token_count, self.model_dim], DATA_TYPE), AllocationType::Global)?;
        let (keys, values) = kv_cache.as_buffer_range_mut().split_at(kv_plane_bytes);
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
        Ok(PreparedPrefixAttention {
            queries,
            kv_cache,
        })
    }

    fn encode_nodes(
        &self,
        residual_input: Allocation<B>,
        residual_state: &mut Allocation<B>,
        prefix_kv: &Allocation<B>,
        node_kv_cache: &mut Allocation<B>,
        node_capacity: u32,
        nodes: &NodeBatch<'_, B>,
        prefix_length: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let attention_input =
            self.pre_attention_norm.encode(&residual_input, 0, nodes.count, Some(residual_state), encoder)?;
        let current_qkv = self.qkv_projection.encode(attention_input, nodes.count, encoder)?;
        let metadata_field_bytes = nodes.count * DataType::U32.size_in_bytes();
        let ancestor_counts = (nodes.metadata, MetadataIdx::AncestorCount as usize * metadata_field_bytes);
        let tree_slot_indices = (nodes.metadata, MetadataIdx::TreeSlot as usize * metadata_field_bytes);
        let mut attention_output =
            encoder.allocate_scratch(size_for_shape(&[nodes.count, self.model_dim], DATA_TYPE))?;
        self.ancestor_attention.encode(
            prefix_kv,
            &*node_kv_cache,
            &current_qkv,
            nodes.ancestor_indices,
            ancestor_counts,
            &mut attention_output,
            nodes.count as u32,
            prefix_length as u32,
            nodes.ancestor_stride as u32,
            node_capacity,
            self.attention_scale,
            encoder,
        );
        // Separate dispatch so the node arena is read-only above; the encoder
        // serializes it after the attention that reads the ancestor slots.
        self.node_cache_write.encode(
            &current_qkv,
            node_kv_cache,
            tree_slot_indices,
            self.model_dim as u32,
            node_capacity,
            (nodes.count * 2 * self.model_dim) as u32,
            encoder,
        );
        self.encode_post_attention(attention_output, residual_state, nodes.count, encoder)
    }

    fn encode_post_attention(
        &self,
        attention_output: Allocation<B>,
        residual_state: &mut Allocation<B>,
        token_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let projected_attention = self.out_projection.encode(attention_output, token_count, encoder)?;
        let mlp_input =
            self.pre_mlp_norm.encode(&projected_attention, 0, token_count, Some(residual_state), encoder)?;
        let mut mlp_hidden = self.up_projection.encode(mlp_input, token_count, encoder)?;
        self.activation.encode(
            None::<&Allocation<B>>,
            &mut mlp_hidden,
            (token_count * self.hidden_dim) as u32,
            crate::backends::common::gpu_types::ActivationType::GELUExact,
            encoder,
        );
        self.down_projection.encode(mlp_hidden, token_count, encoder)
    }
}

fn nodes_from_packed_tree(packed_tree: &[u32]) -> Vec<ProposalNode> {
    assert!(
        packed_tree.len().is_multiple_of(TreeIdx::COUNT),
        "packed tree must contain {} equal-length fields",
        TreeIdx::COUNT
    );
    let tree_slot_count = packed_tree.len() / TreeIdx::COUNT;
    let field = |field: TreeIdx, slot: usize| packed_tree[field as usize * tree_slot_count + slot];
    let mut slot_to_node_index = vec![usize::MAX; tree_slot_count];
    let mut nodes: Vec<ProposalNode> = Vec::with_capacity(tree_slot_count);
    for slot in 0..tree_slot_count {
        if field(TreeIdx::Valid, slot) == 0 {
            continue;
        }
        let parent_slot = field(TreeIdx::ParentSlot, slot) as i32;
        let parent = (parent_slot >= 0).then(|| {
            let parent = slot_to_node_index[parent_slot as usize];
            assert_ne!(parent, usize::MAX, "tree slot {slot} names padding slot {parent_slot} as its parent");
            parent
        });
        let index = nodes.len();
        slot_to_node_index[slot] = index;
        if let Some(parent) = parent {
            nodes[parent].child_indices.push(index);
        }
        nodes.push(ProposalNode {
            token_id: field(TreeIdx::TokenId, slot),
            depth: field(TreeIdx::Depth, slot) as usize,
            child_indices: Vec::new(),
        });
    }
    nodes
}
