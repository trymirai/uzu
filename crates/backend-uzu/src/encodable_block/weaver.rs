use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AsBufferRangeMut, Backend, Encoder, Kernels,
        gpu_types::weaver::{
            CANDIDATES_MAX, FRONTIER_MAX_SLOTS, FRONTIER_NO_WINNER, FrontierIdx, MetadataIdx, TreeIdx,
        },
        kernel::{
            AncestorAttentionKernel, AttentionPrepareKernel, WeaverFrontierInsertChildrenKernel,
            WeaverFrontierSelectKernel, WeaverTopChildrenKernel,
        },
    },
    config::{
        activation::{AnyActivation, silu::SiLU},
        linear::LinearConfig,
        mlp::{AnyMLPConfig, dense_mlp::DenseMLPConfig},
        rope::AnyRoPEConfig,
        weaver::WeaverConfig,
    },
    data_type::DataType,
    encodable_block::{
        embedding::{Embedding, EmbeddingError},
        linear::{Linear, LinearBlockError},
        mixer::attention::{
            AttentionStateType,
            core::{AttentionCoreEncodeArguments, AttentionCoreNewArguments, AttentionCores},
            rope::PrecalculatedRoPE,
        },
        mlp::{Mlp, MlpBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

const DATA_TYPE: DataType = DataType::BF16;
const ROPE_DATA_TYPE: DataType = DataType::F32;

fn weaver_mlp_config(linear_config: LinearConfig) -> AnyMLPConfig {
    AnyMLPConfig::DenseMLPConfig(DenseMLPConfig::unclipped(
        linear_config,
        AnyActivation::SiLU(SiLU::new(1.0)),
        true,
        true,
    ))
}

struct TopKChildren<B: Backend> {
    token_ids: Allocation<B>,
    logprobs: Allocation<B>,
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

struct NodeExpansionKvCache<B: Backend> {
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
    pub ids: &'a Allocation<B>,
    pub logits: &'a Allocation<B>,
    pub depth_count: usize,
    pub candidates_per_depth: usize,
}

pub struct WeaverInputs<'a, B: Backend> {
    pub target_hidden: &'a Allocation<B>,
    pub draft_hidden: &'a Allocation<B>,
    pub target_embedding: &'a Embedding<B>,
    pub candidates: CandidatePool<'a, B>,
    pub depth_seeds: &'a [u64],
    pub root_token_id: u32,
}

pub struct TreeShape {
    pub budget: usize,
    pub frontier_width: usize,
    pub children_per_node: usize,
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
    rope_config: AnyRoPEConfig,
    top_children: <B::Kernels as Kernels>::WeaverTopChildrenKernel,
    frontier_select: <B::Kernels as Kernels>::WeaverFrontierSelectKernel,
    frontier_insert_children: <B::Kernels as Kernels>::WeaverFrontierInsertChildrenKernel,
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
    out_projection: Box<dyn Linear<B>>,

    // MLP
    pre_mlp_norm: Normalization<B>,
    mlp: Box<dyn Mlp<B>>,

    // Geometry
    attention_scale: f32,
    model_dim: usize,
    num_heads: usize,
    head_dim: usize,
    max_depth: usize,
}

#[derive(Debug, Error)]
pub enum WeaverNewError<B: Backend> {
    #[error("parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("mlp error: {0}")]
    Mlp(#[from] MlpBlockError<B>),
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
    #[error("rope head_dim {actual} does not match model_dim / num_heads = {expected}")]
    InvalidRopeHeadDim {
        expected: usize,
        actual: usize,
    },
    #[error("rope max_sequence_length {actual} is too small for max_depth {max_depth} (needs {max_depth} + 1)")]
    InvalidRopeLength {
        max_depth: usize,
        actual: usize,
    },
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
        let head_dim = config.model_dim / config.num_heads;
        if *config.rope_config.head_dim() != head_dim {
            return Err(WeaverNewError::InvalidRopeHeadDim {
                expected: head_dim,
                actual: *config.rope_config.head_dim(),
            });
        }
        if *config.rope_config.max_sequence_length() <= config.max_depth {
            return Err(WeaverNewError::InvalidRopeLength {
                max_depth: config.max_depth,
                actual: *config.rope_config.max_sequence_length(),
            });
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
            .map(|index| WeaverLayer::new(context, config, index > 0, &layer_parameters.subtree(&index.to_string())?))
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
        let top_children =
            <B::Kernels as Kernels>::WeaverTopChildrenKernel::new(context).map_err(WeaverNewError::Backend)?;
        let frontier_select =
            <B::Kernels as Kernels>::WeaverFrontierSelectKernel::new(context).map_err(WeaverNewError::Backend)?;
        let frontier_insert_children = <B::Kernels as Kernels>::WeaverFrontierInsertChildrenKernel::new(context)
            .map_err(WeaverNewError::Backend)?;
        Ok(Self {
            token_embedding_norm,
            token_embedding_projection,
            hidden_state_norm,
            hidden_state_projection,
            layers,
            readout_norm,
            readout_query_projection,
            rope_config: config.rope_config.clone(),
            top_children,
            frontier_select,
            frontier_insert_children,
            model_dim: config.model_dim,
            target_model_dim: config.target_model_dim,
            max_depth: config.max_depth,
        })
    }

    pub fn encode_tree(
        &self,
        inputs: WeaverInputs<'_, B>,
        shape: TreeShape,
        encoder: &mut Encoder<B>,
    ) -> Result<EncodedWeaverTree<B>, WeaverEncodeError<B>> {
        encoder.push_debug_group("dflash weaver");
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

        let rope = self.precalculate_rope(encoder).map_err(WeaverEncodeError::Backend)?;

        let prefix_cache = self
            .encode_prefix(inputs.target_hidden, inputs.draft_hidden, &rope, lookahead_count, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let mut node_cache =
            self.create_node_expansion_kv_cache(tree_slot_count, encoder).map_err(WeaverEncodeError::Backend)?;

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
            if batch_start_slot + batch_node_count == tree_slot_count {
                break; // packed-tree slots filled
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
            let top_children = self.encode_node_expansion(
                &prefix_cache,
                &nodes,
                &candidates,
                &mut node_cache,
                &rope,
                shape.children_per_node,
                inputs.target_embedding,
                encoder,
            )?;
            self.frontier_insert_children.encode(
                &packed_tree,
                &node_metadata,
                &node_valid,
                &top_children.token_ids,
                &top_children.logprobs,
                &mut frontier,
                frontier_capacity as u32,
                tree_slot_count as u32,
                batch_node_count as u32,
                shape.children_per_node as u32,
                encoder,
            );
            batch_start_slot += batch_node_count;
        }
        encoder.pop_debug_group();
        Ok(EncodedWeaverTree {
            packed_tree,
        })
    }

    fn precalculate_rope(
        &self,
        encoder: &mut Encoder<B>,
    ) -> Result<PrecalculatedRoPE<B>, B::Error> {
        let positions = (0..=self.max_depth).collect::<Box<[_]>>();
        PrecalculatedRoPE::precalculate(&self.rope_config, &positions, encoder)
    }

    fn encode_prefix(
        &self,
        target_hidden: &Allocation<B>,
        draft_hidden: &Allocation<B>,
        rope: &PrecalculatedRoPE<B>,
        lookahead_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<PrefixKvCache<B>, B::Error> {
        assert!((1..=self.max_depth).contains(&lookahead_count));
        let token_count = lookahead_count + 1;
        let hidden_row_bytes = self.target_model_dim * DATA_TYPE.size_in_bytes();

        let mut prefix_hidden =
            encoder.allocate_scratch(size_for_shape(&[token_count, self.target_model_dim], DATA_TYPE))?;
        encoder.encode_copy(target_hidden, 0..hidden_row_bytes, &mut prefix_hidden, 0..hidden_row_bytes);
        encoder.encode_copy(
            draft_hidden,
            hidden_row_bytes..token_count * hidden_row_bytes,
            &mut prefix_hidden,
            hidden_row_bytes..token_count * hidden_row_bytes,
        );
        let normalized_prefix = self.hidden_state_norm.encode(&prefix_hidden, 0, token_count, None, encoder)?;
        let mut residual_input = self.hidden_state_projection.encode(normalized_prefix, token_count, encoder)?;
        let (last_layer, preceding_layers) = self.layers.split_last().expect("Weaver must have at least one layer");
        let mut residual_state = encoder.allocate_scratch(residual_input.size())?;
        let mut prefix_layers = Vec::with_capacity(self.layers.len());
        for layer in preceding_layers {
            let PrefixLayerOutput {
                mlp_delta,
                kv_cache,
            } = layer.encode_prefix_tokens(residual_input, &mut residual_state, rope, token_count, encoder)?;
            prefix_layers.push(kv_cache);
            residual_input = mlp_delta;
        }
        prefix_layers.push(
            last_layer
                .encode_prefix_attention(&residual_input, &mut residual_state, rope, token_count, encoder)?
                .kv_cache,
        );
        Ok(PrefixKvCache {
            layers: prefix_layers.into_boxed_slice(),
            length: token_count,
        })
    }

    fn create_node_expansion_kv_cache(
        &self,
        capacity: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<NodeExpansionKvCache<B>, B::Error> {
        assert!(capacity > 0, "Weaver node capacity must be positive");
        let kv_size = size_for_shape(&[2, capacity, self.model_dim], DATA_TYPE);
        let layers =
            (0..self.layers.len()).map(|_| encoder.allocate_scratch(kv_size)).collect::<Result<Box<[_]>, _>>()?;
        Ok(NodeExpansionKvCache {
            layers,
            capacity: capacity as u32,
        })
    }

    fn encode_node_expansion(
        &self,
        prefix_cache: &PrefixKvCache<B>,
        nodes: &NodeBatch<'_, B>,
        candidates: &CandidateBatch<'_, B>,
        node_cache: &mut NodeExpansionKvCache<B>,
        rope: &PrecalculatedRoPE<B>,
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

        let mut residual_state = encoder.allocate_scratch(residual_input.size()).map_err(WeaverEncodeError::Backend)?;
        let node_capacity = node_cache.capacity;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            residual_input = layer
                .encode_node_batch(
                    residual_input,
                    &mut residual_state,
                    &prefix_cache.layers[layer_index],
                    &mut node_cache.layers[layer_index],
                    node_capacity,
                    nodes,
                    rope,
                    prefix_cache.length,
                    encoder,
                )
                .map_err(WeaverEncodeError::Backend)?;
        }

        self.encode_child_selection(
            &residual_input,
            &mut residual_state,
            nodes,
            candidates,
            children_per_node,
            target_embedding,
            encoder,
        )
    }

    fn encode_child_selection(
        &self,
        residual_output: &Allocation<B>,
        residual_state: &mut Allocation<B>,
        nodes: &NodeBatch<'_, B>,
        candidates: &CandidateBatch<'_, B>,
        children_per_node: usize,
        target_embedding: &Embedding<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<TopKChildren<B>, WeaverEncodeError<B>> {
        let normalized_output = self
            .readout_norm
            .encode(residual_output, 0, nodes.count, Some(residual_state), encoder)
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
        let mut logprobs = encoder
            .allocate_scratch(size_for_shape(&[nodes.count, children_per_node], DataType::F32))
            .map_err(WeaverEncodeError::Backend)?;
        self.top_children.encode(
            &logit_residuals,
            candidates.logits,
            candidates.ids,
            candidates.depth_seeds,
            nodes.metadata,
            &mut token_ids,
            &mut logprobs,
            nodes.count as u32,
            candidates.candidates_per_node as u32,
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

impl<B: Backend> WeaverLayer<B> {
    fn new(
        context: &B::Context,
        config: &WeaverConfig,
        add_to_residual: bool,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, WeaverNewError<B>> {
        let WeaverConfig {
            model_dim,
            hidden_dim,
            num_heads,
            max_depth,
            ref norm_config,
            ref linear_config,
            ..
        } = *config;
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
            <B::Kernels as Kernels>::AttentionPrepareKernel::new(context, DATA_TYPE, ROPE_DATA_TYPE, true, true)
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
        let (mlp, up_input_hadamard_factors) = <dyn Mlp<B>>::new(
            &weaver_mlp_config(linear_config.clone()),
            model_dim,
            hidden_dim,
            context,
            &parameter_tree.subtree("mlp")?,
            DATA_TYPE,
        )?;
        assert!(up_input_hadamard_factors.is_none(), "Weaver MLP does not support input Hadamard factors");
        let ancestor_attention =
            <B::Kernels as Kernels>::AncestorAttentionKernel::new(context, head_dim as u32, num_heads as u32)
                .map_err(WeaverNewError::Backend)?;
        Ok(Self {
            qkv_projection,
            out_projection,
            prefix_attention,
            attention_prepare,
            pre_attention_norm,
            pre_mlp_norm,
            mlp,
            ancestor_attention,
            attention_scale,
            model_dim,
            num_heads,
            head_dim,
            max_depth,
        })
    }

    fn encode_prefix_tokens(
        &self,
        residual_input: Allocation<B>,
        residual_state: &mut Allocation<B>,
        rope: &PrecalculatedRoPE<B>,
        token_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<PrefixLayerOutput<B>, B::Error> {
        let PreparedPrefixAttention {
            queries,
            kv_cache,
        } = self.encode_prefix_attention(&residual_input, residual_state, rope, token_count, encoder)?;
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

    fn encode_prefix_attention(
        &self,
        residual_input: &Allocation<B>,
        residual_state: &mut Allocation<B>,
        rope: &PrecalculatedRoPE<B>,
        token_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<PreparedPrefixAttention<B>, B::Error> {
        let attention_input =
            self.pre_attention_norm.encode(residual_input, 0, token_count, Some(residual_state), encoder)?;
        let qkv = self.qkv_projection.encode(attention_input, token_count, encoder)?;
        let mut queries =
            encoder.allocate_scratch(size_for_shape(&[self.num_heads, token_count, self.head_dim], DATA_TYPE))?;
        let kv_plane_bytes = size_for_shape(&[token_count, self.model_dim], DATA_TYPE);
        let mut kv_cache = encoder.allocate_scratch(size_for_shape(&[2, token_count, self.model_dim], DATA_TYPE))?;
        let (keys, values) = kv_cache.as_buffer_range_mut().split_at(kv_plane_bytes);
        self.attention_prepare.encode(
            &qkv,
            &mut queries,
            Some(keys),
            Some(values),
            Some(&rope.cosines),
            Some(&rope.sines),
            self.num_heads as u32,
            Some(self.num_heads as u32),
            self.head_dim as u32,
            Some(rope.dim as u32),
            Some(0),
            token_count as u32,
            encoder,
        );
        Ok(PreparedPrefixAttention {
            queries,
            kv_cache,
        })
    }

    fn encode_node_batch(
        &self,
        residual_input: Allocation<B>,
        residual_state: &mut Allocation<B>,
        prefix_kv: &Allocation<B>,
        node_kv_cache: &mut Allocation<B>,
        node_capacity: u32,
        nodes: &NodeBatch<'_, B>,
        rope: &PrecalculatedRoPE<B>,
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
            node_kv_cache,
            &current_qkv,
            &rope.cosines,
            &rope.sines,
            nodes.metadata,
            nodes.ancestor_indices,
            ancestor_counts,
            tree_slot_indices,
            &mut attention_output,
            nodes.count as u32,
            prefix_length as u32,
            nodes.ancestor_stride as u32,
            node_capacity,
            self.max_depth as u32,
            self.attention_scale,
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
        self.mlp.encode(mlp_input, token_count, encoder)
    }
}

fn nodes_from_packed_tree(packed_tree: &[u32]) -> Vec<ProposalNode> {
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
