use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::weaver::{
            CANDIDATES_MAX, FRONTIER_MAX_SLOTS, FRONTIER_NO_WINNER, FrontierIdx, MetadataIdx, TreeIdx,
        },
        kernel::{
            AncestorAttentionKernel, WeaverFrontierInsertChildrenKernel, WeaverFrontierSelectKernel,
            WeaverTopChildrenKernel,
        },
    },
    config::{rope::AnyRoPEConfig, weaver::WeaverConfig},
    data_type::DataType,
    encodable_block::{
        dflash::TopKCandidates,
        embedding::{Embedding, EmbeddingError},
        linear::{Linear, LinearBlockError},
        mixer::attention::{AttentionStateType, core::AttentionCoreEncodeArguments, rope::PrecalculatedRoPE},
        mlp::MlpBlockError,
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        weaver_layer::{PreparedPrefixAttention, WeaverLayer},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

pub(crate) const DATA_TYPE: DataType = DataType::BF16;
pub(crate) const ROPE_DATA_TYPE: DataType = DataType::F32;

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
        let packed_tree = &self.packed_tree.copyout::<u32>();

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
        target_hidden: &Allocation<B>,
        draft_hidden: &Allocation<B>,
        target_embedding: &Embedding<B>,
        candidates: &TopKCandidates<B>,
        depth_seeds: &[u64],
        root_token_id: u32,
        shape: TreeShape,
        encoder: &mut Encoder<B>,
    ) -> Result<EncodedWeaverTree<B>, WeaverEncodeError<B>> {
        encoder.push_debug_group("weaver tree");

        let tree_slot_count = shape.budget + 1;
        let frontier_capacity = tree_slot_count * shape.children_per_node;
        let lookahead_count = self.max_depth.min(candidates.rows);
        let ancestor_stride = self.max_depth;
        if shape.budget == 0
            || shape.frontier_width == 0
            || shape.children_per_node == 0
            || shape.children_per_node > candidates.candidates_per_row
            || frontier_capacity > FRONTIER_MAX_SLOTS
            || lookahead_count == 0
            || depth_seeds.len() != self.max_depth
        {
            return Err(WeaverEncodeError::InvalidTreeInput);
        }

        let rope_positions = (0..=self.max_depth).collect::<Box<[_]>>();
        let rope = PrecalculatedRoPE::precalculate(&self.rope_config, &rope_positions, encoder)
            .map_err(WeaverEncodeError::Backend)?;

        // Prefix pass: run the target token and the draft lookahead rows through
        // every layer, collecting per-layer KV caches.
        let prefix_length = lookahead_count + 1;
        let hidden_row_bytes = self.target_model_dim * DATA_TYPE.size_in_bytes();
        let mut prefix_hidden = encoder
            .allocate_scratch(size_for_shape(&[prefix_length, self.target_model_dim], DATA_TYPE))
            .map_err(WeaverEncodeError::Backend)?;
        encoder.encode_copy(target_hidden, 0..hidden_row_bytes, &mut prefix_hidden, 0..hidden_row_bytes);
        encoder.encode_copy(
            draft_hidden,
            hidden_row_bytes..prefix_length * hidden_row_bytes,
            &mut prefix_hidden,
            hidden_row_bytes..prefix_length * hidden_row_bytes,
        );
        let normalized_prefix = self
            .hidden_state_norm
            .encode(&prefix_hidden, 0, prefix_length, None, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let mut residual_input = self
            .hidden_state_projection
            .encode(normalized_prefix, prefix_length, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let (last_layer, preceding_layers) = self.layers.split_last().expect("Weaver must have at least one layer");
        let mut residual_state = encoder.allocate_scratch(residual_input.size()).map_err(WeaverEncodeError::Backend)?;
        let mut prefix_kv_layers = Vec::with_capacity(self.layers.len());
        for layer in preceding_layers {
            let PreparedPrefixAttention {
                queries,
                kv_cache,
            } = layer
                .encode_prefix_attention(&residual_input, &mut residual_state, &rope, prefix_length, encoder)
                .map_err(WeaverEncodeError::Backend)?;
            let state_type = AttentionStateType::Full {
                length: 0,
            };
            let kv_plane_bytes = size_for_shape(&[prefix_length, self.model_dim], DATA_TYPE);
            let attention_output = layer
                .prefix_attention
                .encode(
                    AttentionCoreEncodeArguments {
                        queries: &queries,
                        keys: &kv_cache,
                        values: (&kv_cache, kv_plane_bytes),
                        suffix_length: prefix_length,
                        trie: None,
                        sinks: None,
                        state_type: &state_type,
                    },
                    encoder,
                )
                .map_err(WeaverEncodeError::Backend)?;
            residual_input = layer
                .encode_post_attention(attention_output, &mut residual_state, prefix_length, encoder)
                .map_err(WeaverEncodeError::Backend)?;
            prefix_kv_layers.push(kv_cache);
        }
        prefix_kv_layers.push(
            last_layer
                .encode_prefix_attention(&residual_input, &mut residual_state, &rope, prefix_length, encoder)
                .map_err(WeaverEncodeError::Backend)?
                .kv_cache,
        );

        // Per-layer KV cache for tree nodes, one slot per packed-tree slot.
        let node_kv_size = size_for_shape(&[2, tree_slot_count, self.model_dim], DATA_TYPE);
        let mut node_kv_layers = (0..self.layers.len())
            .map(|_| encoder.allocate_scratch(node_kv_size))
            .collect::<Result<Vec<_>, _>>()
            .map_err(WeaverEncodeError::Backend)?;

        let mut tree_init = vec![0u32; TreeIdx::COUNT * tree_slot_count];
        for slot in 0..tree_slot_count {
            tree_init[TreeIdx::ParentSlot as usize * tree_slot_count + slot] = FRONTIER_NO_WINNER;
        }
        tree_init[TreeIdx::TokenId as usize * tree_slot_count] = root_token_id;
        tree_init[TreeIdx::Valid as usize * tree_slot_count] = 1;

        let mut packed_tree = encoder.allocate_constant_from_slice(&tree_init).map_err(WeaverEncodeError::Backend)?;
        let mut frontier = encoder
            .allocate_constant_from_slice(&vec![0u32; FrontierIdx::COUNT * frontier_capacity])
            .map_err(WeaverEncodeError::Backend)?;
        let mut slot_ancestors = encoder
            .allocate_constant_from_slice(&vec![0u32; tree_slot_count * ancestor_stride])
            .map_err(WeaverEncodeError::Backend)?;

        let mut initial_node_token_ids = vec![0u32; shape.frontier_width];
        initial_node_token_ids[0] = root_token_id;
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
            .allocate_constant_from_slice(&vec![0u32; shape.frontier_width * candidates.candidates_per_row])
            .map_err(WeaverEncodeError::Backend)?;
        let mut node_candidate_logits = encoder
            .allocate_constant_from_slice(&vec![0.0f32; shape.frontier_width * candidates.candidates_per_row])
            .map_err(WeaverEncodeError::Backend)?;
        let depth_seeds_buffer =
            encoder.allocate_constant_from_slice(depth_seeds).map_err(WeaverEncodeError::Backend)?;

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
                    &candidates.ids,
                    &candidates.logits,
                    &mut node_candidate_ids,
                    &mut node_candidate_logits,
                    frontier_capacity as u32,
                    tree_slot_count as u32,
                    batch_node_count as u32,
                    batch_start_slot as u32,
                    ancestor_stride as u32,
                    self.max_depth as u32,
                    lookahead_count as u32,
                    candidates.rows as u32,
                    candidates.candidates_per_row as u32,
                    encoder,
                );
            }
            if batch_start_slot + batch_node_count == tree_slot_count {
                break; // packed-tree slots filled
            }
            let (batch_candidate_ids, batch_candidate_logits) = if batch_start_slot == 0 {
                (&candidates.ids, &candidates.logits)
            } else {
                (&node_candidate_ids, &node_candidate_logits)
            };

            // Node expansion: embed the batch's tokens, run every layer against
            // the prefix KV and each node's ancestors, then pick its children.
            let token_embedding = target_embedding.encode_lookup(&node_token_ids, batch_node_count, encoder)?;
            let normalized_embedding = self
                .token_embedding_norm
                .encode(&token_embedding, 0, batch_node_count, None, encoder)
                .map_err(WeaverEncodeError::Backend)?;
            let mut residual_input = self
                .token_embedding_projection
                .encode(normalized_embedding, batch_node_count, encoder)
                .map_err(WeaverEncodeError::Backend)?;
            let mut residual_state =
                encoder.allocate_scratch(residual_input.size()).map_err(WeaverEncodeError::Backend)?;
            let metadata_field_bytes = batch_node_count * DataType::U32.size_in_bytes();
            for (layer_index, layer) in self.layers.iter().enumerate() {
                let attention_input = layer
                    .pre_attention_norm
                    .encode(&residual_input, 0, batch_node_count, Some(&mut residual_state), encoder)
                    .map_err(WeaverEncodeError::Backend)?;
                let current_qkv = layer
                    .qkv_projection
                    .encode(attention_input, batch_node_count, encoder)
                    .map_err(WeaverEncodeError::Backend)?;
                let mut attention_output = encoder
                    .allocate_scratch(size_for_shape(&[batch_node_count, self.model_dim], DATA_TYPE))
                    .map_err(WeaverEncodeError::Backend)?;
                layer.ancestor_attention.encode(
                    &prefix_kv_layers[layer_index],
                    &mut node_kv_layers[layer_index],
                    &current_qkv,
                    &rope.cosines,
                    &rope.sines,
                    &node_metadata,
                    &node_ancestor_indices,
                    (&node_metadata, MetadataIdx::AncestorCount as usize * metadata_field_bytes),
                    (&node_metadata, MetadataIdx::TreeSlot as usize * metadata_field_bytes),
                    &mut attention_output,
                    batch_node_count as u32,
                    prefix_length as u32,
                    ancestor_stride as u32,
                    tree_slot_count as u32,
                    layer.max_depth as u32,
                    layer.attention_scale,
                    encoder,
                );
                residual_input = layer
                    .encode_post_attention(attention_output, &mut residual_state, batch_node_count, encoder)
                    .map_err(WeaverEncodeError::Backend)?;
            }

            let normalized_output = self
                .readout_norm
                .encode(&residual_input, 0, batch_node_count, Some(&mut residual_state), encoder)
                .map_err(WeaverEncodeError::Backend)?;
            let query = self
                .readout_query_projection
                .encode(normalized_output, batch_node_count, encoder)
                .map_err(WeaverEncodeError::Backend)?;
            let logit_residuals = target_embedding.encode_readout_sparse(
                &query,
                batch_candidate_ids,
                batch_node_count,
                candidates.candidates_per_row,
                encoder,
            )?;
            let mut child_token_ids = encoder
                .allocate_scratch(size_for_shape(&[batch_node_count, shape.children_per_node], DataType::U32))
                .map_err(WeaverEncodeError::Backend)?;
            let mut child_logprobs = encoder
                .allocate_scratch(size_for_shape(&[batch_node_count, shape.children_per_node], DataType::F32))
                .map_err(WeaverEncodeError::Backend)?;
            self.top_children.encode(
                &logit_residuals,
                batch_candidate_logits,
                batch_candidate_ids,
                &depth_seeds_buffer,
                &node_metadata,
                &mut child_token_ids,
                &mut child_logprobs,
                batch_node_count as u32,
                candidates.candidates_per_row as u32,
                shape.children_per_node as u32,
                target_embedding.vocab_size() as u32,
                encoder,
            );

            self.frontier_insert_children.encode(
                &packed_tree,
                &node_metadata,
                &node_valid,
                &child_token_ids,
                &child_logprobs,
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
}
