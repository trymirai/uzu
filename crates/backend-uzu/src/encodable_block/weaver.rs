use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder, Kernels,
        gpu_types::weaver::{CANDIDATES_MAX, MetadataIdx},
        kernel::{
            ActivationKernel, AttentionLastQueryKernel, AttentionPrepareKernel, TensorAddBiasKernel,
            TensorAddSwapKernel, WeaverNodeCacheWriteKernel, WeaverTopChildrenKernel,
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
        normalization::{Normalization, NormalizationNewError, PostLayerScalar},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

const DATA_TYPE: DataType = DataType::BF16;

pub(crate) struct TopKChildren<B: Backend> {
    pub(crate) token_ids: Allocation<B>,
    pub(crate) logprobs: Allocation<B>,
}

struct PrefixLayerOutput<B: Backend> {
    hidden: Allocation<B>,
    kv: Allocation<B>,
}

pub(crate) struct WeaverPrefixKvCache<B: Backend> {
    layer_kv: Box<[Allocation<B>]>,
    token_count: usize,
}

pub(crate) struct WeaverNodeKvCache<B: Backend> {
    layer_kv: Box<[Allocation<B>]>,
    node_capacity: u32,
}

pub(crate) struct WeaverStepBatch<'a, B: Backend> {
    pub node_count: usize,
    pub candidates_per_node: usize,
    pub ancestor_stride: usize,
    pub node_token_ids: &'a Allocation<B>,
    pub candidate_ids: &'a Allocation<B>,
    pub candidate_logits: &'a Allocation<B>,
    pub ancestor_indices: &'a Allocation<B>,
    pub node_metadata: &'a Allocation<B>,
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
    last_query_attention: <B::Kernels as Kernels>::AttentionLastQueryKernel,
    node_cache_write: <B::Kernels as Kernels>::WeaverNodeCacheWriteKernel,
    out_projection: Box<dyn Linear<B>>,
    residual_add: <B::Kernels as Kernels>::TensorAddSwapKernel,

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
}

impl<B: Backend> Weaver<B> {
    pub(crate) fn create_node_kv_cache(
        &self,
        capacity: usize,
        context: &B::Context,
    ) -> Result<WeaverNodeKvCache<B>, WeaverEncodeError<B>> {
        debug_assert!(capacity > 0 && capacity <= u32::MAX as usize);
        let capacity = capacity as u32;
        let kv_size = size_for_shape(&[2, capacity as usize, self.model_dim], DATA_TYPE);
        let layer_kv = (0..self.blocks.len())
            .map(|_| context.create_allocation(kv_size, AllocationType::Global).map_err(WeaverEncodeError::Backend))
            .collect::<Result<Box<[_]>, _>>()?;
        Ok(WeaverNodeKvCache {
            layer_kv,
            node_capacity: capacity,
        })
    }

    pub(crate) fn new(
        context: &B::Context,
        config: &WeaverConfig,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, WeaverNewError<B>> {
        if config.num_heads == 0 || !config.model_dim.is_multiple_of(config.num_heads) {
            return Err(WeaverNewError::InvalidHeadConfig);
        }
        if config.candidate_pool_size == 0 || config.candidate_pool_size > CANDIDATES_MAX {
            return Err(WeaverNewError::InvalidCandidatePoolSize(config.candidate_pool_size));
        }
        let token_embedding_norm = Normalization::new(
            config.target_embedding_dim,
            None,
            false,
            false,
            PostLayerScalar::None,
            DATA_TYPE,
            &config.norm_config,
            &parameter_tree.subtree("embedding_norm")?,
            context,
        )?;
        let hidden_state_norm = Normalization::new(
            config.target_model_dim,
            None,
            false,
            false,
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
                    &config.norm_config,
                    &blocks_tree.subtree(&index.to_string())?,
                )
            })
            .collect::<Result<Box<[_]>, WeaverNewError<B>>>()?;
        let readout_norm = Normalization::new(
            config.model_dim,
            None,
            false,
            false,
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
            model_dim: config.model_dim,
            target_model_dim: config.target_model_dim,
            max_depth: config.max_depth,
        })
    }

    pub(crate) fn build_prefix(
        &self,
        target_hidden: &Allocation<B>,
        lookaheads: &Allocation<B>,
        lookahead_offset: usize,
        lookahead_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<WeaverPrefixKvCache<B>, WeaverEncodeError<B>> {
        let token_count = lookahead_count + 1;
        assert!(lookahead_count <= self.max_depth);
        let row_bytes = self.target_model_dim * DATA_TYPE.size_in_bytes();
        assert!(target_hidden.size() >= row_bytes);
        assert!(lookaheads.size() >= (lookahead_offset + lookahead_count) * row_bytes);

        let mut input = encoder
            .allocate_scratch(size_for_shape(&[token_count, self.target_model_dim], DATA_TYPE))
            .map_err(WeaverEncodeError::Backend)?;
        encoder.encode_copy(target_hidden, 0..row_bytes, &mut input, 0..row_bytes);
        if lookahead_count > 0 {
            encoder.encode_copy(
                lookaheads,
                lookahead_offset * row_bytes..(lookahead_offset + lookahead_count) * row_bytes,
                &mut input,
                row_bytes..token_count * row_bytes,
            );
        }
        let normalized =
            self.hidden_state_norm.encode(&input, 0, token_count, None, encoder).map_err(WeaverEncodeError::Backend)?;
        let mut hidden = self
            .hidden_state_projection
            .encode(normalized, token_count, encoder)
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

        let mut layer_kv = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let PrefixLayerOutput {
                hidden: next_hidden,
                kv,
            } = block.encode_prefix(hidden, token_count, encoder).map_err(WeaverEncodeError::Backend)?;
            layer_kv.push(kv);
            hidden = next_hidden;
        }
        drop(input);
        drop(hidden);
        Ok(WeaverPrefixKvCache {
            layer_kv: layer_kv.into_boxed_slice(),
            token_count,
        })
    }

    pub(crate) fn encode_step_batch(
        &self,
        prefix: &WeaverPrefixKvCache<B>,
        batch: &WeaverStepBatch<'_, B>,
        node_cache: &mut WeaverNodeKvCache<B>,
        children_per_node: usize,
        target_embedding: &Embedding<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<TopKChildren<B>, WeaverEncodeError<B>> {
        let rows = batch.node_count;
        let candidates = batch.candidates_per_node;
        assert!(rows > 0);
        assert!(candidates > 0 && candidates <= CANDIDATES_MAX);
        assert!(children_per_node > 0 && children_per_node <= candidates);
        assert!(batch.ancestor_stride > 0);
        let word = DataType::U32.size_in_bytes();
        debug_assert!(batch.node_metadata.size() >= MetadataIdx::COUNT * rows * word);
        debug_assert!(batch.node_token_ids.size() >= rows * word);
        debug_assert!(batch.ancestor_indices.size() >= rows * batch.ancestor_stride * word);

        let token_embedding = target_embedding.encode_lookup(batch.node_token_ids, rows, encoder)?;
        let embedding_normalized = self
            .token_embedding_norm
            .encode(&token_embedding, 0, rows, None, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let mut hidden = self
            .token_embedding_projection
            .encode(embedding_normalized, rows, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        self.node_position_add.encode(
            None::<&Allocation<B>>,
            &self.position_embeddings,
            Some(batch.node_metadata),
            &mut hidden,
            self.model_dim as u32,
            (rows * self.model_dim) as u32,
            encoder,
        );

        let node_capacity = node_cache.node_capacity;
        for (layer_index, block) in self.blocks.iter().enumerate() {
            hidden = block
                .encode_step(
                    hidden,
                    &prefix.layer_kv[layer_index],
                    &mut node_cache.layer_kv[layer_index],
                    node_capacity,
                    batch,
                    prefix.token_count,
                    encoder,
                )
                .map_err(WeaverEncodeError::Backend)?;
        }

        self.encode_step_output(&hidden, batch, children_per_node, target_embedding, encoder)
    }

    /// Outputs must outlive the batch.
    fn encode_step_output(
        &self,
        hidden: &Allocation<B>,
        batch: &WeaverStepBatch<'_, B>,
        children_per_node: usize,
        target_embedding: &Embedding<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<TopKChildren<B>, WeaverEncodeError<B>> {
        let (rows, candidates) = (batch.node_count, batch.candidates_per_node);
        let output_normalized =
            self.readout_norm.encode(hidden, 0, rows, None, encoder).map_err(WeaverEncodeError::Backend)?;
        let query = self
            .readout_query_projection
            .encode(output_normalized, rows, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        let residual_logits =
            target_embedding.encode_readout_sparse(&query, batch.candidate_ids, rows, candidates, encoder)?;
        let mut token_ids = encoder
            .allocate_scratch(size_for_shape(&[rows, children_per_node], DataType::U32))
            .map_err(WeaverEncodeError::Backend)?;
        let mut logprobs = encoder
            .allocate_scratch(size_for_shape(&[rows, children_per_node], DataType::F32))
            .map_err(WeaverEncodeError::Backend)?;
        self.top_children.encode(
            &residual_logits,
            batch.candidate_logits,
            batch.candidate_ids,
            &mut token_ids,
            &mut logprobs,
            rows as u32,
            candidates as u32,
            children_per_node as u32,
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
            false,
            false,
            PostLayerScalar::None,
            DATA_TYPE,
            norm_config,
            &parameter_tree.subtree("pre_attention_norm")?,
            context,
        )?;
        let pre_mlp_norm = Normalization::new(
            model_dim,
            None,
            false,
            false,
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
        let residual_add =
            <B::Kernels as Kernels>::TensorAddSwapKernel::new(context, DATA_TYPE).map_err(WeaverNewError::Backend)?;
        let last_query_attention =
            <B::Kernels as Kernels>::AttentionLastQueryKernel::new(context, head_dim as u32, num_heads as u32)
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
            residual_add,
            last_query_attention,
            node_cache_write,
            attention_scale,
            model_dim,
            hidden_dim,
            num_heads,
            head_dim,
        })
    }

    fn encode_step(
        &self,
        hidden: Allocation<B>,
        prefix_kv: &Allocation<B>,
        node_kv_cache: &mut Allocation<B>,
        node_capacity: u32,
        batch: &WeaverStepBatch<'_, B>,
        prefix_length: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let normalized = self.pre_attention_norm.encode(&hidden, 0, batch.node_count, None, encoder)?;
        let current_qkv = self.qkv_projection.encode(normalized, batch.node_count, encoder)?;
        let metadata_lane_bytes = batch.node_count * DataType::U32.size_in_bytes();
        let ancestor_counts = (batch.node_metadata, MetadataIdx::AncestorCount as usize * metadata_lane_bytes);
        let tree_slots = (batch.node_metadata, MetadataIdx::TreeSlot as usize * metadata_lane_bytes);
        let mut attention_output =
            encoder.allocate_scratch(size_for_shape(&[batch.node_count, self.model_dim], DATA_TYPE))?;
        self.last_query_attention.encode(
            prefix_kv,
            &*node_kv_cache,
            &current_qkv,
            batch.ancestor_indices,
            ancestor_counts,
            &mut attention_output,
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
        attention_output = self.out_projection.encode(attention_output, batch.node_count, encoder)?;
        let mut residual = hidden;
        self.residual_add.encode(
            &mut residual,
            &mut attention_output,
            (batch.node_count * self.model_dim) as u32,
            encoder,
        );
        let normalized = self.pre_mlp_norm.encode(&residual, 0, batch.node_count, None, encoder)?;
        let mut mlp_hidden = self.up_projection.encode(normalized, batch.node_count, encoder)?;
        self.activation.encode(
            None::<&Allocation<B>>,
            &mut mlp_hidden,
            (batch.node_count * self.hidden_dim) as u32,
            crate::backends::common::gpu_types::ActivationType::GELUExact,
            encoder,
        );
        let mut mlp_output = self.down_projection.encode(mlp_hidden, batch.node_count, encoder)?;
        self.residual_add.encode(&mut mlp_output, &mut residual, (batch.node_count * self.model_dim) as u32, encoder);
        Ok(residual)
    }

    fn encode_prefix(
        &self,
        hidden: Allocation<B>,
        token_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<PrefixLayerOutput<B>, B::Error> {
        let normalized = self.pre_attention_norm.encode(&hidden, 0, token_count, None, encoder)?;
        let qkv = self.qkv_projection.encode(normalized, token_count, encoder)?;
        let mut queries =
            encoder.allocate_scratch(size_for_shape(&[self.num_heads, token_count, self.head_dim], DATA_TYPE))?;
        let kv_plane_bytes = size_for_shape(&[token_count, self.model_dim], DATA_TYPE);
        let mut kv = encoder
            .context()
            .create_allocation(size_for_shape(&[2, token_count, self.model_dim], DATA_TYPE), AllocationType::Global)?;
        let (keys, values) = kv.split_at_mut(kv_plane_bytes);
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
        let state_type = AttentionStateType::Full {
            length: 0,
        };
        let attention = self.prefix_attention.encode(
            AttentionCoreEncodeArguments {
                queries: &queries,
                keys: &kv,
                values: (&kv, kv_plane_bytes),
                suffix_length: token_count,
                trie: None,
                sinks: None,
                state_type: &state_type,
            },
            encoder,
        )?;
        let mut attention = self.out_projection.encode(attention, token_count, encoder)?;
        let mut residual = hidden;
        self.residual_add.encode(&mut residual, &mut attention, (token_count * self.model_dim) as u32, encoder);
        let normalized = self.pre_mlp_norm.encode(&residual, 0, token_count, None, encoder)?;
        let mut mlp = self.up_projection.encode(normalized, token_count, encoder)?;
        self.activation.encode(
            None::<&Allocation<B>>,
            &mut mlp,
            (token_count * self.hidden_dim) as u32,
            crate::backends::common::gpu_types::ActivationType::GELUExact,
            encoder,
        );
        let mut output = self.down_projection.encode(mlp, token_count, encoder)?;
        self.residual_add.encode(&mut output, &mut residual, (token_count * self.model_dim) as u32, encoder);
        Ok(PrefixLayerOutput {
            hidden: residual,
            kv,
        })
    }
}
