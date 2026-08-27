use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context, Kernels,
        gpu_types::weaver::{
            CANDIDATES_MAX, FRONTIER_MAX_SLOTS, FRONTIER_MAX_WIDTH, FrontierIdx, MetadataIdx, TreeIdx,
        },
        kernel::{
            AncestorAttentionKernel, AttentionArguments, AttentionKernel, WeaverFrontierInsertChildrenKernel,
            WeaverFrontierSelectKernel, WeaverTopChildrenKernel, radix_top_k_small::RadixTopKSmall,
        },
    },
    config::{rope::AnyRoPEConfig, weaver::WeaverConfig},
    data_type::DataType,
    encodable_block::{
        embedding::{Embedding, EmbeddingError},
        linear::{Linear, LinearBlockError},
        mixer::attention::{KVCacheView, rope::PrecalculatedRoPE},
        mlp::MlpBlockError,
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        weaver_layer::{PreparedPrefixAttention, WeaverLayer},
    },
    parameters::ParameterTree,
};

pub const DATA_TYPE: DataType = DataType::BF16;
pub const ROPE_DATA_TYPE: DataType = DataType::F32;

pub struct WeaverTreeShape {
    pub tree_budget: u32,
    pub max_depth: u32,
    pub dflash_depth: u32,
    pub rounds: u32,
    pub expand_per_round: u32,
    pub expand_width: u32,
}

impl WeaverTreeShape {
    fn slot_count(&self) -> u32 {
        1 + self.rounds.saturating_sub(1) * self.expand_per_round
    }
}

pub struct EncodedWeaverTree<B: Backend> {
    packed_tree: B::GlobalBuffer,
    frontier: B::GlobalBuffer,
}

pub struct ProposalNode {
    pub token_id: u32,
    pub depth: u32,
    pub logprob: f32,
    pub child_indices: Vec<usize>,
}

impl<B: Backend> EncodedWeaverTree<B> {
    pub fn read_nodes(self) -> Vec<ProposalNode> {
        let packed_tree = &self.packed_tree.copyout::<u32>();
        let frontier = &self.frontier.copyout::<u32>();

        let tree_slot_count = packed_tree.len() / TreeIdx::COUNT;
        let frontier_capacity = frontier.len() / FrontierIdx::COUNT;
        let tree_field = |field: TreeIdx, slot: usize| packed_tree[field as usize * tree_slot_count + slot];
        let frontier_field = |field: FrontierIdx, slot: usize| frontier[field as usize * frontier_capacity + slot];

        let mut slot_to_index = vec![usize::MAX; tree_slot_count];
        let mut nodes: Vec<ProposalNode> = Vec::new();
        for slot in 0..tree_slot_count {
            if tree_field(TreeIdx::Valid, slot) == 0 {
                continue;
            }
            let parent_slot = tree_field(TreeIdx::ParentSlot, slot) as i32;
            let parent = (parent_slot >= 0).then(|| {
                let parent = slot_to_index[parent_slot as usize];
                assert_ne!(parent, usize::MAX, "tree slot {slot} names padding slot {parent_slot} as its parent");
                parent
            });
            let index = nodes.len();
            slot_to_index[slot] = index;
            if let Some(parent) = parent {
                nodes[parent].child_indices.push(index);
            }
            nodes.push(ProposalNode {
                token_id: tree_field(TreeIdx::TokenId, slot),
                depth: tree_field(TreeIdx::Depth, slot),
                logprob: f32::from_bits(tree_field(TreeIdx::EdgeLogprobBits, slot)),
                child_indices: Vec::new(),
            });
        }
        for slot in 0..frontier_capacity {
            if frontier_field(FrontierIdx::Active, slot) == 0 {
                continue;
            }
            let parent_slot = frontier_field(FrontierIdx::ParentSlot, slot) as usize;
            let parent = slot_to_index[parent_slot];
            assert_ne!(parent, usize::MAX, "frontier slot {slot} names padding slot {parent_slot} as its parent");
            let index = nodes.len();
            nodes[parent].child_indices.push(index);
            nodes.push(ProposalNode {
                token_id: frontier_field(FrontierIdx::TokenId, slot),
                depth: frontier_field(FrontierIdx::Depth, slot),
                logprob: f32::from_bits(frontier_field(FrontierIdx::EdgeLogprobBits, slot)),
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
    top_k: <B::Kernels as Kernels>::RadixTopKSmall,
    top_children: <B::Kernels as Kernels>::WeaverTopChildrenKernel,
    frontier_select: <B::Kernels as Kernels>::WeaverFrontierSelectKernel,
    frontier_insert_children: <B::Kernels as Kernels>::WeaverFrontierInsertChildrenKernel,
    model_dim: u32,
    target_model_dim: u32,
    max_depth: u32,
    candidate_pool_size: u32,
}

#[derive(Debug, Error)]
pub enum WeaverNewError<B: Backend> {
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
    InvalidCandidatePoolSize(u32),
    #[error("rope head_dim {actual} does not match model_dim / num_heads = {expected}")]
    InvalidRopeHeadDim {
        expected: u32,
        actual: u32,
    },
    #[error("rope max_sequence_length {actual} is too small for max_depth {max_depth} (needs {max_depth} + 1)")]
    InvalidRopeLength {
        max_depth: u32,
        actual: u32,
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
        vocab_size: u32,
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
            &parameter_tree.subtree("embedding_norm"),
            context,
        )?;
        let hidden_state_norm = Normalization::new(
            config.target_model_dim,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            DATA_TYPE,
            &config.norm_config,
            &parameter_tree.subtree("hidden_state_norm"),
            context,
        )?;
        let token_embedding_projection = <dyn Linear<B>>::new(
            config.target_embedding_dim,
            [config.model_dim],
            true,
            context,
            DATA_TYPE,
            &parameter_tree.subtree("embedding_projection"),
        )?;
        let layer_parameters = parameter_tree.subtree("blocks");
        let layers = (0..config.num_layers)
            .map(|index| WeaverLayer::new(context, config, index > 0, &layer_parameters.subtree(&index.to_string())))
            .collect::<Result<Box<[_]>, WeaverNewError<B>>>()?;
        let readout_norm = Normalization::new(
            config.model_dim,
            None,
            ShortcutMode::Add,
            PostLayerScalar::None,
            DATA_TYPE,
            &config.norm_config,
            &parameter_tree.subtree("output_norm"),
            context,
        )?;
        let hidden_state_projection = <dyn Linear<B>>::new(
            config.target_model_dim,
            [config.model_dim],
            true,
            context,
            DATA_TYPE,
            &parameter_tree.subtree("hidden_state_projection"),
        )?;
        let readout_query_projection = <dyn Linear<B>>::new(
            config.model_dim,
            [config.target_model_dim],
            false,
            context,
            DATA_TYPE,
            &parameter_tree.subtree("query_projection"),
        )?;
        let top_k =
            <B::Kernels as Kernels>::RadixTopKSmall::new(context, vocab_size).map_err(WeaverNewError::Backend)?;
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
            top_k,
            top_children,
            frontier_select,
            frontier_insert_children,
            model_dim: config.model_dim,
            target_model_dim: config.target_model_dim,
            max_depth: config.max_depth,
            candidate_pool_size: config.candidate_pool_size,
        })
    }

    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    fn encode_prefix(
        &self,
        target_hidden: impl BufferRef<Backend = B>,
        draft_hidden: impl BufferRef<Backend = B>,
        rope: &PrecalculatedRoPE<B>,
        depth: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Vec<B::ScratchBuffer>, WeaverEncodeError<B>> {
        command_buffer.push_debug_group("weaver prefix");

        let hidden_row_bytes = size_for_shape(&[self.target_model_dim], DATA_TYPE);
        let mut prefix_hidden = command_buffer
            .allocate_scratch_for_shape(&[depth, self.target_model_dim], DATA_TYPE)
            .map_err(WeaverEncodeError::Backend)?;
        command_buffer
            .encode_copy(target_hidden.subrange(..hidden_row_bytes), prefix_hidden.subrange_mut(..hidden_row_bytes));
        command_buffer.encode_copy(
            draft_hidden.subrange(hidden_row_bytes..depth as usize * hidden_row_bytes),
            prefix_hidden.subrange_mut(hidden_row_bytes..),
        );
        let normalized_prefix = self
            .hidden_state_norm
            .encode(&prefix_hidden, 0, depth, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(WeaverEncodeError::Backend)?;
        let mut residual_input = self
            .hidden_state_projection
            .encode(normalized_prefix, depth, command_buffer)
            .map_err(WeaverEncodeError::Backend)?;
        let (last_layer, preceding_layers) = self.layers.split_last().expect("Weaver must have at least one layer");
        let mut residual_state =
            command_buffer.allocate_scratch(residual_input.size()).map_err(WeaverEncodeError::Backend)?;
        let mut prefix_kv_layers = Vec::with_capacity(self.layers.len());
        for layer in preceding_layers {
            let PreparedPrefixAttention {
                queries,
                kv_cache,
            } = layer
                .encode_prefix_attention(&residual_input, &mut residual_state, rope, depth, command_buffer)
                .map_err(WeaverEncodeError::Backend)?;
            let cache = KVCacheView::full(0);
            let kv_plane_bytes = size_for_shape(&[depth, self.model_dim], DATA_TYPE);
            let attention_output = layer
                .prefix_attention
                .encode(
                    AttentionArguments {
                        queries: &queries,
                        keys: &kv_cache,
                        values: kv_cache.subrange(kv_plane_bytes..),
                        suffix_length: depth,
                        trie: None::<&B::ConstantBuffer>,
                        sinks: None,
                        cache,
                    },
                    command_buffer,
                )
                .map_err(WeaverEncodeError::Backend)?;
            residual_input = layer
                .encode_post_attention(attention_output, &mut residual_state, depth, command_buffer)
                .map_err(WeaverEncodeError::Backend)?;
            prefix_kv_layers.push(kv_cache);
        }
        prefix_kv_layers.push(
            last_layer
                .encode_prefix_attention(&residual_input, &mut residual_state, rope, depth, command_buffer)
                .map_err(WeaverEncodeError::Backend)?
                .kv_cache,
        );

        command_buffer.pop_debug_group();

        Ok(prefix_kv_layers)
    }

    fn encode_step<Ids: BufferRef<Backend = B>, Logits: BufferRef<Backend = B>>(
        &self,
        target_embedding: &Embedding<B>,
        mut prefix_kv_layers: impl Iterator<Item = impl BufferRef<Backend = B>>,
        rope: &PrecalculatedRoPE<B>,
        mut node_kv_layers: impl Iterator<Item = impl BufferMut<Backend = B>>,
        mut packed_tree: impl BufferMut<Backend = B>,
        mut frontier: impl BufferMut<Backend = B>,
        slot_ancestors: impl BufferMut<Backend = B>,
        mut node_token_ids: impl BufferMut<Backend = B>,
        mut node_metadata: impl BufferMut<Backend = B>,
        mut node_ancestor_indices: impl BufferMut<Backend = B>,
        mut node_valid: impl BufferMut<Backend = B>,
        mut node_candidate_ids: impl BufferMut<Backend = B, Buffer = Ids::Buffer>,
        mut node_candidate_logits: impl BufferMut<Backend = B, Buffer = Logits::Buffer>,
        depth_seeds_buffer: impl BufferRef<Backend = B>,
        candidate_ids: Ids,
        candidate_logits: Logits,
        shape: &WeaverTreeShape,
        batch_node_count: u32,
        batch_start_slot: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), WeaverEncodeError<B>> {
        let tree_slot_count = shape.slot_count();
        let ancestor_stride = self.max_depth;
        let frontier_capacity = tree_slot_count * shape.expand_width;

        if batch_start_slot > 0 {
            self.frontier_select.encode(
                frontier.reborrow(),
                packed_tree.reborrow(),
                slot_ancestors,
                node_token_ids.reborrow(),
                node_metadata.reborrow(),
                node_ancestor_indices.reborrow(),
                node_valid.reborrow(),
                candidate_ids,
                candidate_logits,
                node_candidate_ids.reborrow(),
                node_candidate_logits.reborrow(),
                frontier_capacity,
                tree_slot_count,
                batch_node_count,
                batch_start_slot,
                ancestor_stride,
                self.max_depth,
                shape.max_depth - 1,
                shape.dflash_depth - 1,
                self.candidate_pool_size,
                command_buffer,
            );
        }
        let (batch_candidate_ids, batch_candidate_logits) = if batch_start_slot == 0 {
            (candidate_ids.subrange(..), candidate_logits.subrange(..))
        } else {
            (node_candidate_ids.as_ref().subrange(..), node_candidate_logits.as_ref().subrange(..))
        };

        // Node expansion: embed the batch's tokens, run every layer against
        // the prefix KV and each node's ancestors, then pick its children.
        let token_embedding =
            target_embedding.encode_lookup(node_token_ids.as_ref(), batch_node_count, command_buffer)?;
        let normalized_embedding = self
            .token_embedding_norm
            .encode(&token_embedding, 0, batch_node_count, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(WeaverEncodeError::Backend)?;
        let mut residual_input = self
            .token_embedding_projection
            .encode(normalized_embedding, batch_node_count, command_buffer)
            .map_err(WeaverEncodeError::Backend)?;
        let mut residual_state =
            command_buffer.allocate_scratch(residual_input.size()).map_err(WeaverEncodeError::Backend)?;
        let node_metadata = node_metadata.as_ref();
        let metadata_field_bytes = size_for_shape(&[batch_node_count], DataType::U32);
        for layer in &self.layers {
            let attention_input = layer
                .pre_attention_norm
                .encode(&residual_input, 0, batch_node_count, Some(&mut residual_state), command_buffer)
                .map_err(WeaverEncodeError::Backend)?;
            let current_qkv = layer
                .qkv_projection
                .encode(attention_input, batch_node_count, command_buffer)
                .map_err(WeaverEncodeError::Backend)?;
            let mut attention_output = command_buffer
                .allocate_scratch_for_shape(&[batch_node_count, self.model_dim], DATA_TYPE)
                .map_err(WeaverEncodeError::Backend)?;
            layer.ancestor_attention.encode(
                prefix_kv_layers.next().expect("missing prefix KV layer"),
                node_kv_layers.next().expect("missing node KV layer"),
                &current_qkv,
                &rope.cosines,
                &rope.sines,
                node_metadata,
                node_ancestor_indices.as_ref(),
                node_metadata.subrange(MetadataIdx::AncestorCount as usize * metadata_field_bytes..),
                node_metadata.subrange(MetadataIdx::TreeSlot as usize * metadata_field_bytes..),
                &mut attention_output,
                batch_node_count,
                shape.dflash_depth,
                ancestor_stride,
                tree_slot_count,
                layer.max_depth,
                layer.attention_scale,
                command_buffer,
            );
            residual_input = layer
                .encode_post_attention(attention_output, &mut residual_state, batch_node_count, command_buffer)
                .map_err(WeaverEncodeError::Backend)?;
        }

        let normalized_output = self
            .readout_norm
            .encode(&residual_input, 0, batch_node_count, Some(&mut residual_state), command_buffer)
            .map_err(WeaverEncodeError::Backend)?;
        let query = self
            .readout_query_projection
            .encode(normalized_output, batch_node_count, command_buffer)
            .map_err(WeaverEncodeError::Backend)?;
        let logit_residuals = target_embedding.encode_readout(
            batch_node_count,
            &query,
            self.candidate_pool_size,
            Some(batch_candidate_ids),
            false,
            command_buffer,
        )?;
        let mut child_token_ids = command_buffer
            .allocate_scratch_for_shape(&[batch_node_count, shape.expand_width], DataType::U32)
            .map_err(WeaverEncodeError::Backend)?;
        let mut child_logprobs = command_buffer
            .allocate_scratch_for_shape(&[batch_node_count, shape.expand_width], DataType::F32)
            .map_err(WeaverEncodeError::Backend)?;
        self.top_children.encode(
            &logit_residuals,
            batch_candidate_logits,
            batch_candidate_ids,
            depth_seeds_buffer,
            node_metadata,
            &mut child_token_ids,
            &mut child_logprobs,
            batch_node_count,
            self.candidate_pool_size,
            shape.expand_width,
            target_embedding.vocab_size(),
            command_buffer,
        );

        self.frontier_insert_children.encode(
            packed_tree.as_ref(),
            node_metadata,
            node_valid.as_ref(),
            &child_token_ids,
            &child_logprobs,
            frontier,
            frontier_capacity,
            tree_slot_count,
            batch_node_count,
            shape.expand_width,
            command_buffer,
        );

        Ok(())
    }

    pub fn encode_tree(
        &self,
        target_hidden: impl BufferRef<Backend = B>,
        draft_hidden: impl BufferRef<Backend = B>,
        target_embedding: &Embedding<B>,
        logits: impl BufferRef<Backend = B>,
        depth_seeds: &[u64],
        root_token_id: u32,
        shape: WeaverTreeShape,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<EncodedWeaverTree<B>, WeaverEncodeError<B>> {
        command_buffer.push_debug_group("weaver tree");

        let tree_slot_count = shape.slot_count();
        let ancestor_stride = self.max_depth;
        if shape.tree_budget == 0
            || shape.rounds == 0
            || shape.max_depth < 2
            || shape.max_depth > self.max_depth + 1
            || shape.dflash_depth < shape.max_depth
            || shape.dflash_depth > self.max_depth + 1
            || shape.expand_per_round == 0
            || shape.expand_per_round > FRONTIER_MAX_WIDTH
            || shape.expand_width == 0
            || shape.expand_width > self.candidate_pool_size
            || tree_slot_count > FRONTIER_MAX_SLOTS / shape.expand_width
            || depth_seeds.len() as u32 != self.max_depth
        {
            return Err(WeaverEncodeError::InvalidTreeInput);
        }
        let frontier_capacity = tree_slot_count * shape.expand_width;
        let pool_depth_count = shape.dflash_depth - 1;

        // Rank the draft logits: the top `candidate_pool_size` tokens per
        // lookahead row form the candidate pool node expansions draw from.
        let vocab_size = target_embedding.vocab_size();
        assert!(
            logits.size() >= size_for_shape(&[pool_depth_count, vocab_size], DATA_TYPE),
            "draft logits do not cover the lookahead rows"
        );
        let mut candidate_ids = command_buffer
            .allocate_scratch_for_shape(&[pool_depth_count, self.candidate_pool_size], DataType::U32)
            .map_err(WeaverEncodeError::Backend)?;
        let mut candidate_logits = command_buffer
            .allocate_scratch_for_shape(&[pool_depth_count, self.candidate_pool_size], DataType::F32)
            .map_err(WeaverEncodeError::Backend)?;
        self.top_k
            .encode(
                logits,
                &mut candidate_ids,
                &mut candidate_logits,
                pool_depth_count,
                self.candidate_pool_size,
                command_buffer,
            )
            .map_err(WeaverEncodeError::Backend)?;

        let rope_positions = (0..=self.max_depth).collect::<Box<[_]>>();
        let rope = PrecalculatedRoPE::precalculate(&self.rope_config, &rope_positions, command_buffer)
            .map_err(WeaverEncodeError::Backend)?;

        let prefix_kv_layers =
            self.encode_prefix(target_hidden, draft_hidden, &rope, shape.dflash_depth, command_buffer)?;

        // Per-layer KV cache for tree nodes, one slot per packed-tree slot.
        let node_kv_size = size_for_shape(&[2, tree_slot_count, self.model_dim], DATA_TYPE);
        let mut node_kv_layers = (0..self.layers.len())
            .map(|_| command_buffer.allocate_scratch(node_kv_size))
            .collect::<Result<Vec<_>, _>>()
            .map_err(WeaverEncodeError::Backend)?;

        let root =
            command_buffer.allocate_constant_from_slice(&[root_token_id, 1u32]).map_err(WeaverEncodeError::Backend)?;
        let word_bytes = DataType::U32.size_in_bytes();
        let tree_field_bytes = size_for_shape(&[tree_slot_count], DataType::U32);
        let mut packed_tree =
            command_buffer.allocate_scratch(TreeIdx::COUNT * tree_field_bytes).map_err(WeaverEncodeError::Backend)?;
        command_buffer.encode_fill(&mut packed_tree, 0);
        let parent_offset = TreeIdx::ParentSlot as usize * tree_field_bytes;
        command_buffer.encode_fill(
            packed_tree.subrange_mut(parent_offset..parent_offset + tree_field_bytes),
            0xff, // FRONTIER_NO_WINNER
        );
        command_buffer.encode_copy(root.subrange(..word_bytes), packed_tree.subrange_mut(..word_bytes));
        let valid_offset = TreeIdx::Valid as usize * tree_field_bytes;
        command_buffer.encode_copy(
            root.subrange(word_bytes..),
            packed_tree.subrange_mut(valid_offset..valid_offset + word_bytes),
        );

        let mut frontier = command_buffer
            .allocate_scratch_for_shape(&[FrontierIdx::COUNT as u32, frontier_capacity], DataType::U32)
            .map_err(WeaverEncodeError::Backend)?;
        command_buffer.encode_fill(&mut frontier, 0);
        let mut slot_ancestors = command_buffer
            .allocate_scratch_for_shape(&[tree_slot_count, ancestor_stride], DataType::U32)
            .map_err(WeaverEncodeError::Backend)?;

        let mut node_token_ids = command_buffer
            .allocate_scratch_for_shape(&[shape.expand_per_round], DataType::U32)
            .map_err(WeaverEncodeError::Backend)?;
        command_buffer.encode_fill(&mut node_token_ids, 0);
        command_buffer.encode_copy(root.subrange(..word_bytes), node_token_ids.subrange_mut(..word_bytes));

        let mut node_metadata = command_buffer
            .allocate_scratch_for_shape(&[MetadataIdx::COUNT as u32, shape.expand_per_round], DataType::U32)
            .map_err(WeaverEncodeError::Backend)?;
        command_buffer.encode_fill(&mut node_metadata, 0);
        let mut node_ancestor_indices = command_buffer
            .allocate_scratch_for_shape(&[shape.expand_per_round, ancestor_stride], DataType::U32)
            .map_err(WeaverEncodeError::Backend)?;
        command_buffer.encode_fill(&mut node_ancestor_indices, 0);
        let mut node_valid = command_buffer
            .allocate_scratch_for_shape(&[shape.expand_per_round], DataType::U32)
            .map_err(WeaverEncodeError::Backend)?;
        command_buffer.encode_fill(&mut node_valid, 0);
        command_buffer.encode_copy(root.subrange(word_bytes..), node_valid.subrange_mut(..word_bytes));

        let mut node_candidate_ids = command_buffer
            .allocate_scratch_for_shape(&[shape.expand_per_round, self.candidate_pool_size], DataType::U32)
            .map_err(WeaverEncodeError::Backend)?;
        let mut node_candidate_logits = command_buffer
            .allocate_scratch_for_shape(&[shape.expand_per_round, self.candidate_pool_size], DataType::F32)
            .map_err(WeaverEncodeError::Backend)?;
        let depth_seeds_buffer =
            command_buffer.allocate_constant_from_slice(depth_seeds).map_err(WeaverEncodeError::Backend)?;

        let mut batch_start_slot = 0;
        for round in 0..shape.rounds {
            let batch_node_count = if round == 0 {
                1
            } else {
                shape.expand_per_round
            };
            command_buffer.push_debug_group("weaver step");
            self.encode_step(
                target_embedding,
                prefix_kv_layers.iter(),
                &rope,
                node_kv_layers.iter_mut(),
                &mut packed_tree,
                &mut frontier,
                &mut slot_ancestors,
                &mut node_token_ids,
                &mut node_metadata,
                &mut node_ancestor_indices,
                &mut node_valid,
                &mut node_candidate_ids,
                &mut node_candidate_logits,
                &depth_seeds_buffer,
                &candidate_ids,
                &candidate_logits,
                &shape,
                batch_node_count,
                batch_start_slot,
                command_buffer,
            )?;
            command_buffer.pop_debug_group();
            batch_start_slot += batch_node_count;
        }

        command_buffer.pop_debug_group();

        let mut packed_tree_readback =
            command_buffer.context().create_buffer(packed_tree.size()).map_err(WeaverEncodeError::Backend)?;
        command_buffer.encode_copy(&packed_tree, &mut packed_tree_readback);
        let mut frontier_readback =
            command_buffer.context().create_buffer(frontier.size()).map_err(WeaverEncodeError::Backend)?;
        command_buffer.encode_copy(&frontier, &mut frontier_readback);

        Ok(EncodedWeaverTree {
            packed_tree: packed_tree_readback,
            frontier: frontier_readback,
        })
    }
}
