use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context, Kernels,
        gpu_types::weaver::{
            CANDIDATES_MAX, FRONTIER_MAX_SLOTS, FRONTIER_MAX_WIDTH, FrontierIdx, MetadataIdx, TreeIdx,
            WeaverSelectionMode::{Children, Frontier, Fused},
        },
        kernel::{
            AncestorAttentionKernel, AttentionArguments, AttentionInput, AttentionKernel, FullPrecisionInput,
            WeaverSelectKernel, radix_top_k_small::RadixTopKSmall,
        },
    },
    config::{rope::AnyRoPEConfig, weaver::WeaverConfig},
    data_type::DataType,
    encodable_block::{
        embedding::{Embedding, EmbeddingError},
        linear::{Linear, LinearBlockError, LinearEncodeError},
        mixer::attention::{KVCacheView, rope::PrecalculatedRoPE},
        mlp::{MlpBlockNewError, MlpEncodeError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        weaver_layer::{PreparedPrefixAttention, WeaverLayer},
    },
    safetensors::ParameterTree,
    utils::size_for_shape::size_for_shape,
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
    pub prune_noise_scale: Option<f32>,
}

impl WeaverTreeShape {
    fn slot_count(&self) -> u32 {
        1 + self.rounds.saturating_sub(1) * self.expand_per_round
    }
}

pub struct EncodedWeaverTree<B: Backend> {
    packed_tree: B::GlobalGpuBuffer,
    frontier: B::GlobalGpuBuffer,
}

pub struct ProposalNode {
    pub token_id: u32,
    pub depth: u32,
    pub logprob: f32,
    pub child_indices: Vec<usize>,
}

impl<B: Backend> EncodedWeaverTree<B> {
    pub async fn read_nodes(
        self,
        context: &B::Context,
    ) -> Result<Vec<ProposalNode>, B::Error> {
        let (packed_tree, frontier) = futures::future::try_join(
            context.copyout::<u32>(&self.packed_tree),
            context.copyout::<u32>(&self.frontier),
        )
        .await?;

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
        Ok(nodes)
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
    top_children: <B::Kernels as Kernels>::WeaverSelectKernel,
    top_children_with_prune_noise: <B::Kernels as Kernels>::WeaverSelectKernel,
    frontier_select: <B::Kernels as Kernels>::WeaverSelectKernel,
    expand_select: <B::Kernels as Kernels>::WeaverSelectKernel,
    model_dim: u32,
    target_model_dim: u32,
    max_depth: u32,
    candidate_pool_size: u32,
}

#[derive(Debug, Error)]
pub enum WeaverNewError<B: Backend> {
    #[error("Attention kernel error: {0}")]
    Attention(#[source] <<B::Kernels as Kernels>::AttentionKernel as AttentionKernel>::Error),
    #[error("linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("mlp error: {0}")]
    Mlp(#[from] MlpBlockNewError<B>),
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
    #[error("Linear error: {0}")]
    Linear(#[from] LinearEncodeError<B>),
    #[error("MLP error: {0}")]
    Mlp(#[from] MlpEncodeError<B>),
    #[error("Attention kernel error: {0}")]
    Attention(#[source] <<B::Kernels as Kernels>::AttentionKernel as AttentionKernel>::Error),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
    #[error("invalid Weaver tree input")]
    InvalidTreeInput,
}

impl<B: Backend> Weaver<B> {
    pub async fn new(
        vocab_size: u32,
        config: &WeaverConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
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
            DATA_TYPE,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            &config.norm_config,
            &parameter_tree.subtree("embedding_norm"),
            context,
        )
        .await?;
        let hidden_state_norm = Normalization::new(
            config.target_model_dim,
            DATA_TYPE,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            &config.norm_config,
            &parameter_tree.subtree("hidden_state_norm"),
            context,
        )
        .await?;
        let token_embedding_projection = <dyn Linear<B>>::new(
            config.target_embedding_dim,
            config.model_dim,
            DATA_TYPE,
            true,
            &parameter_tree.subtree("embedding_projection"),
            context,
        )
        .await?;
        let layer_parameters = parameter_tree.subtree("blocks");
        let mut layers = Vec::with_capacity(config.num_layers as usize);
        for index in 0..config.num_layers {
            layers.push(
                WeaverLayer::new(index > 0, config, &layer_parameters.subtree(&index.to_string()), context).await?,
            );
        }
        let readout_norm = Normalization::new(
            config.model_dim,
            DATA_TYPE,
            None,
            ShortcutMode::Add,
            PostLayerScalar::None,
            &config.norm_config,
            &parameter_tree.subtree("output_norm"),
            context,
        )
        .await?;
        let hidden_state_projection = <dyn Linear<B>>::new(
            config.target_model_dim,
            config.model_dim,
            DATA_TYPE,
            true,
            &parameter_tree.subtree("hidden_state_projection"),
            context,
        )
        .await?;
        let readout_query_projection = <dyn Linear<B>>::new(
            config.model_dim,
            config.target_model_dim,
            DATA_TYPE,
            false,
            &parameter_tree.subtree("query_projection"),
            context,
        )
        .await?;
        let top_k =
            <B::Kernels as Kernels>::RadixTopKSmall::new(context, vocab_size).map_err(WeaverNewError::Backend)?;
        let top_children = <B::Kernels as Kernels>::WeaverSelectKernel::new(context, 256, Children, false)
            .map_err(WeaverNewError::Backend)?;
        let top_children_with_prune_noise =
            <B::Kernels as Kernels>::WeaverSelectKernel::new(context, 256, Children, true)
                .map_err(WeaverNewError::Backend)?;
        let frontier_select = <B::Kernels as Kernels>::WeaverSelectKernel::new(context, 256, Frontier, false)
            .map_err(WeaverNewError::Backend)?;
        let expand_select = <B::Kernels as Kernels>::WeaverSelectKernel::new(context, 1024, Fused, false)
            .map_err(WeaverNewError::Backend)?;
        Ok(Self {
            token_embedding_norm,
            token_embedding_projection,
            hidden_state_norm,
            hidden_state_projection,
            layers: layers.into_boxed_slice(),
            readout_norm,
            readout_query_projection,
            rope_config: config.rope_config.clone(),
            top_k,
            top_children,
            top_children_with_prune_noise,
            frontier_select,
            expand_select,
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
        let command_buffer = &mut *command_buffer.debug_group("weaver prefix");

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
        let mut residual_input = self.hidden_state_projection.encode(normalized_prefix, depth, command_buffer)?;
        let (last_layer, preceding_layers) = self.layers.split_last().expect("Weaver must have at least one layer");
        let mut residual_state =
            command_buffer.allocate_scratch(residual_input.size()).map_err(WeaverEncodeError::Backend)?;
        let mut prefix_kv_layers = Vec::with_capacity(self.layers.len());
        for layer in preceding_layers {
            let PreparedPrefixAttention {
                queries,
                kv_cache,
            } = layer.encode_prefix_attention(&residual_input, &mut residual_state, rope, depth, command_buffer)?;
            let cache = KVCacheView::full(0);
            let kv_plane_bytes = size_for_shape(&[depth, self.model_dim], DATA_TYPE);
            let attention_output = layer
                .prefix_attention
                .encode(
                    AttentionInput::FullPrecision(FullPrecisionInput {
                        queries: &queries,
                        keys: &kv_cache,
                        values: kv_cache.subrange(kv_plane_bytes..),
                    }),
                    AttentionArguments {
                        suffix_length: depth,
                        trie: None::<&B::ConstantBuffer>,
                        sinks: None,
                        cache,
                        gate: None::<&B::ConstantBuffer>,
                    },
                    command_buffer,
                )
                .map_err(WeaverEncodeError::Attention)?;
            residual_input =
                layer.encode_post_attention(attention_output, &mut residual_state, depth, command_buffer)?;
            prefix_kv_layers.push(kv_cache);
        }
        prefix_kv_layers.push(
            last_layer
                .encode_prefix_attention(&residual_input, &mut residual_state, rope, depth, command_buffer)?
                .kv_cache,
        );

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
        mut slot_ancestors: impl BufferMut<Backend = B>,
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

        let fuse_select = shape.expand_per_round <= 4 && shape.prune_noise_scale.is_none();
        if batch_start_slot > 0 && !fuse_select {
            self.frontier_select.encode(
                None::<&B::ScratchBuffer>,
                None::<&B::ScratchBuffer>,
                frontier.reborrow(),
                packed_tree.reborrow(),
                Some(slot_ancestors.reborrow()),
                Some(node_token_ids.reborrow()),
                node_metadata.reborrow(),
                Some(node_ancestor_indices.reborrow()),
                node_valid.reborrow(),
                candidate_ids,
                candidate_logits,
                node_candidate_ids.reborrow(),
                node_candidate_logits.reborrow(),
                0,
                0,
                self.candidate_pool_size,
                shape.expand_width,
                target_embedding.vocab_size(),
                frontier_capacity,
                tree_slot_count,
                batch_node_count,
                batch_start_slot,
                ancestor_stride,
                shape.max_depth - 1,
                shape.dflash_depth - 1,
                None,
                command_buffer,
            );
        }
        let batch_candidate_ids = if batch_start_slot == 0 {
            candidate_ids.subrange(..)
        } else {
            node_candidate_ids.as_ref().subrange(..)
        };

        // Node expansion: embed the batch's tokens, run every layer against
        // the prefix KV and each node's ancestors, then pick its children.
        let token_embedding =
            target_embedding.encode_lookup(node_token_ids.as_ref(), batch_node_count, command_buffer)?;
        let normalized_embedding = self
            .token_embedding_norm
            .encode(&token_embedding, 0, batch_node_count, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(WeaverEncodeError::Backend)?;
        let mut residual_input =
            self.token_embedding_projection.encode(normalized_embedding, batch_node_count, command_buffer)?;
        let mut residual_state =
            command_buffer.allocate_scratch(residual_input.size()).map_err(WeaverEncodeError::Backend)?;
        let node_metadata_ref = node_metadata.as_ref();
        let metadata_field_bytes = size_for_shape(&[batch_node_count], DataType::U32);
        for layer in &self.layers {
            let input = layer
                .pre_attention_norm
                .encode(&residual_input, 0, batch_node_count, Some(&mut residual_state), command_buffer)
                .map_err(WeaverEncodeError::Backend)?;
            let current_qkv = layer.qkv_projection.encode(input, batch_node_count, command_buffer)?;
            let mut attention_output = command_buffer
                .allocate_scratch_for_shape(&[batch_node_count, self.model_dim], DATA_TYPE)
                .map_err(WeaverEncodeError::Backend)?;
            layer.ancestor_attention.encode(
                prefix_kv_layers.next().expect("missing prefix KV layer"),
                node_kv_layers.next().expect("missing node KV layer"),
                &current_qkv,
                &rope.cosines,
                &rope.sines,
                node_metadata_ref,
                node_ancestor_indices.as_ref(),
                node_metadata_ref.subrange(MetadataIdx::AncestorCount as usize * metadata_field_bytes..),
                node_metadata_ref.subrange(MetadataIdx::TreeSlot as usize * metadata_field_bytes..),
                &mut attention_output,
                batch_node_count,
                shape.dflash_depth,
                ancestor_stride,
                tree_slot_count,
                layer.max_depth,
                layer.attention_scale,
                command_buffer,
            );
            residual_input =
                layer.encode_post_attention(attention_output, &mut residual_state, batch_node_count, command_buffer)?;
        }

        let normalized_output = self
            .readout_norm
            .encode(&residual_input, 0, batch_node_count, Some(&mut residual_state), command_buffer)
            .map_err(WeaverEncodeError::Backend)?;
        let query = self.readout_query_projection.encode(normalized_output, batch_node_count, command_buffer)?;
        let logit_residuals = target_embedding.encode_readout(
            &query,
            batch_node_count,
            self.candidate_pool_size,
            Some(batch_candidate_ids),
            false,
            command_buffer,
        )?;
        let next_batch_start = batch_start_slot + batch_node_count;
        let fuse_next = fuse_select && next_batch_start < tree_slot_count;
        let select = if fuse_next {
            &self.expand_select
        } else if shape.prune_noise_scale.is_some() {
            &self.top_children_with_prune_noise
        } else {
            &self.top_children
        };
        select.encode(
            Some(&logit_residuals),
            Some(depth_seeds_buffer),
            frontier.reborrow(),
            packed_tree.reborrow(),
            fuse_next.then_some(slot_ancestors.reborrow()),
            fuse_next.then_some(node_token_ids.reborrow()),
            node_metadata.reborrow(),
            fuse_next.then_some(node_ancestor_indices.reborrow()),
            node_valid.reborrow(),
            candidate_ids,
            candidate_logits,
            node_candidate_ids.reborrow(),
            node_candidate_logits.reborrow(),
            batch_node_count,
            u32::from(batch_start_slot == 0),
            self.candidate_pool_size,
            shape.expand_width,
            target_embedding.vocab_size(),
            frontier_capacity,
            tree_slot_count,
            if fuse_next {
                shape.expand_per_round
            } else {
                0
            },
            next_batch_start,
            ancestor_stride,
            shape.max_depth - 1,
            shape.dflash_depth - 1,
            shape.prune_noise_scale,
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
        let (packed_tree, frontier) = {
            let command_buffer = &mut *command_buffer.debug_group("weaver tree");

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
                || shape.prune_noise_scale.is_some_and(|scale| !(scale > 0.0 && scale.is_finite()))
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

            let root = command_buffer
                .allocate_constant_from_slice(&[root_token_id, 1u32])
                .map_err(WeaverEncodeError::Backend)?;
            let word_bytes = DataType::U32.size_in_bytes();
            let tree_field_bytes = size_for_shape(&[tree_slot_count], DataType::U32);
            let mut packed_tree = command_buffer
                .allocate_scratch(TreeIdx::COUNT * tree_field_bytes)
                .map_err(WeaverEncodeError::Backend)?;
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
                let command_buffer = &mut *command_buffer.debug_group("weaver step");
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
                batch_start_slot += batch_node_count;
            }

            (packed_tree, frontier)
        };

        let mut packed_tree_readback =
            command_buffer.context().create_buffer_gpu(packed_tree.size()).map_err(WeaverEncodeError::Backend)?;
        command_buffer.encode_copy(&packed_tree, &mut packed_tree_readback);
        let mut frontier_readback =
            command_buffer.context().create_buffer_gpu(frontier.size()).map_err(WeaverEncodeError::Backend)?;
        command_buffer.encode_copy(&frontier, &mut frontier_readback);

        Ok(EncodedWeaverTree {
            packed_tree: packed_tree_readback,
            frontier: frontier_readback,
        })
    }
}
