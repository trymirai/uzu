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

pub(crate) struct WeaverPrefix<B: Backend> {
    layer_qkv: Box<[Allocation<B>]>,
    pub(crate) length: usize,
}

pub(crate) struct WeaverStepBatch<'a, B: Backend> {
    pub row_count: usize,
    pub candidate_count: usize,
    pub ancestor_stride: usize,
    pub parent_token_ids: &'a Allocation<B>,
    pub candidate_ids: &'a Allocation<B>,
    pub candidate_scores: &'a Allocation<B>,
    pub ancestor_indices: &'a Allocation<B>,
    pub node_metadata: &'a Allocation<B>,
}

pub(crate) struct WeaverNodeState<B: Backend> {
    layer_qkv: Box<[Allocation<B>]>,
    capacity: u32,
}

pub(crate) struct Weaver<B: Backend> {
    embedding_norm: Normalization<B>,
    hidden_state_norm: Normalization<B>,
    embedding_projection: Box<dyn Linear<B>>,
    blocks: Box<[WeaverBlock<B>]>,
    output_norm: Normalization<B>,
    hidden_state_projection: Box<dyn Linear<B>>,
    query_projection: Box<dyn Linear<B>>,
    position_embeddings: Allocation<B>,
    position_add: <B::Kernels as Kernels>::TensorAddBiasKernel,
    indexed_position_add: <B::Kernels as Kernels>::TensorAddBiasKernel,
    top_children: <B::Kernels as Kernels>::WeaverTopChildrenKernel,
    model_dim: usize,
    target_model_dim: usize,
    max_depth: usize,
}

struct WeaverBlock<B: Backend> {
    qkv_projection: Box<dyn Linear<B>>,
    out_projection: Box<dyn Linear<B>>,
    prefix_attention: AttentionCores<B>,
    attention_prepare: <B::Kernels as Kernels>::AttentionPrepareKernel,
    pre_attention_norm: Normalization<B>,
    pre_mlp_norm: Normalization<B>,
    up_projection: Box<dyn Linear<B>>,
    down_projection: Box<dyn Linear<B>>,
    activation: <B::Kernels as Kernels>::ActivationKernel,
    residual_add: <B::Kernels as Kernels>::TensorAddSwapKernel,
    last_query_attention: <B::Kernels as Kernels>::AttentionLastQueryKernel,
    node_cache_write: <B::Kernels as Kernels>::WeaverNodeCacheWriteKernel,
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
    pub(crate) fn create_node_state(
        &self,
        capacity: usize,
        context: &B::Context,
    ) -> Result<WeaverNodeState<B>, WeaverEncodeError<B>> {
        debug_assert!(capacity > 0 && capacity <= u32::MAX as usize);
        let capacity = capacity as u32;
        let qkv_size = size_for_shape(&[capacity as usize, 3, self.model_dim], DATA_TYPE);
        let layer_qkv = (0..self.blocks.len())
            .map(|_| context.create_allocation(qkv_size, AllocationType::Global).map_err(WeaverEncodeError::Backend))
            .collect::<Result<Box<[_]>, _>>()?;
        Ok(WeaverNodeState {
            layer_qkv,
            capacity,
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
        let embedding_norm = Normalization::new(
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
        let embedding_projection = <dyn Linear<B>>::new(
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
        let output_norm = Normalization::new(
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
        let query_projection = <dyn Linear<B>>::new(
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
        let position_add =
            <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, DATA_TYPE, DataType::F32, true, false)
                .map_err(WeaverNewError::Backend)?;
        let indexed_position_add =
            <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, DATA_TYPE, DataType::F32, true, true)
                .map_err(WeaverNewError::Backend)?;
        let top_children =
            <B::Kernels as Kernels>::WeaverTopChildrenKernel::new(context).map_err(WeaverNewError::Backend)?;
        Ok(Self {
            embedding_norm,
            hidden_state_norm,
            embedding_projection,
            blocks,
            output_norm,
            hidden_state_projection,
            query_projection,
            position_embeddings,
            position_add,
            indexed_position_add,
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
    ) -> Result<WeaverPrefix<B>, WeaverEncodeError<B>> {
        let length = lookahead_count + 1;
        assert!(lookahead_count <= self.max_depth);
        let row_bytes = self.target_model_dim * DATA_TYPE.size_in_bytes();
        assert!(target_hidden.size() >= row_bytes);
        assert!(lookaheads.size() >= (lookahead_offset + lookahead_count) * row_bytes);

        let context = encoder.context();
        let mut input = encoder
            .allocate_scratch(size_for_shape(&[length, self.target_model_dim], DATA_TYPE))
            .map_err(WeaverEncodeError::Backend)?;
        encoder.encode_copy(target_hidden, 0..row_bytes, &mut input, 0..row_bytes);
        if lookahead_count > 0 {
            encoder.encode_copy(
                lookaheads,
                lookahead_offset * row_bytes..(lookahead_offset + lookahead_count) * row_bytes,
                &mut input,
                row_bytes..length * row_bytes,
            );
        }
        let normalized =
            self.hidden_state_norm.encode(&input, 0, length, None, encoder).map_err(WeaverEncodeError::Backend)?;
        let mut hidden =
            self.hidden_state_projection.encode(normalized, length, encoder).map_err(WeaverEncodeError::Backend)?;
        let position_elements = lookahead_count * self.model_dim;
        self.position_add.encode(
            None::<&Allocation<B>>,
            &self.position_embeddings,
            None::<&Allocation<B>>,
            (&mut hidden, self.model_dim * DATA_TYPE.size_in_bytes()),
            position_elements as u32,
            position_elements as u32,
            encoder,
        );

        let mut layer_qkv = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (next_hidden, qkv) =
                block.encode_prefix(hidden, length, encoder).map_err(WeaverEncodeError::Backend)?;
            let mut cached_qkv =
                context.create_allocation(qkv.size(), AllocationType::Global).map_err(WeaverEncodeError::Backend)?;
            encoder.encode_copy(&qkv, .., &mut cached_qkv, ..);
            layer_qkv.push(cached_qkv);
            hidden = next_hidden;
        }
        drop(input);
        drop(hidden);
        Ok(WeaverPrefix {
            layer_qkv: layer_qkv.into_boxed_slice(),
            length,
        })
    }

    pub(crate) fn encode_step_batch(
        &self,
        prefix: &WeaverPrefix<B>,
        input: &WeaverStepBatch<'_, B>,
        state: &mut WeaverNodeState<B>,
        children: usize,
        target_embedding: &Embedding<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Allocation<B>), WeaverEncodeError<B>> {
        let rows = input.row_count;
        let candidates = input.candidate_count;
        assert!(rows > 0);
        assert!(candidates > 0 && candidates <= CANDIDATES_MAX);
        assert!(children > 0 && children <= candidates);
        assert!(input.ancestor_stride > 0);
        let word = DataType::U32.size_in_bytes();
        debug_assert!(input.node_metadata.size() >= MetadataIdx::COUNT * rows * word);
        debug_assert!(input.parent_token_ids.size() >= rows * word);
        debug_assert!(input.ancestor_indices.size() >= rows * input.ancestor_stride * word);

        let token_embedding = target_embedding.encode_lookup(input.parent_token_ids, rows, encoder)?;
        let embedding_normalized =
            self.embedding_norm.encode(&token_embedding, 0, rows, None, encoder).map_err(WeaverEncodeError::Backend)?;
        let mut current = self
            .embedding_projection
            .encode(embedding_normalized, rows, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        self.indexed_position_add.encode(
            None::<&Allocation<B>>,
            &self.position_embeddings,
            Some(input.node_metadata),
            &mut current,
            self.model_dim as u32,
            (rows * self.model_dim) as u32,
            encoder,
        );

        let node_capacity = state.capacity;
        for (layer_index, block) in self.blocks.iter().enumerate() {
            current = block
                .encode_step(
                    current,
                    &prefix.layer_qkv[layer_index],
                    &mut state.layer_qkv[layer_index],
                    node_capacity,
                    input,
                    prefix.length,
                    encoder,
                )
                .map_err(WeaverEncodeError::Backend)?;
        }

        self.encode_step_output(&current, input, children, target_embedding, encoder)
    }

    /// Outputs must outlive the batch.
    fn encode_step_output(
        &self,
        current: &Allocation<B>,
        step: &WeaverStepBatch<'_, B>,
        children: usize,
        target_embedding: &Embedding<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Allocation<B>), WeaverEncodeError<B>> {
        let (rows, candidates) = (step.row_count, step.candidate_count);
        let output_normalized =
            self.output_norm.encode(current, 0, rows, None, encoder).map_err(WeaverEncodeError::Backend)?;
        let query =
            self.query_projection.encode(output_normalized, rows, encoder).map_err(WeaverEncodeError::Backend)?;
        let candidate_logits =
            target_embedding.encode_readout_sparse(&query, step.candidate_ids, rows, candidates, encoder)?;
        let mut token_ids = encoder
            .allocate_scratch(size_for_shape(&[rows, children], DataType::U32))
            .map_err(WeaverEncodeError::Backend)?;
        let mut model_logprobs = encoder
            .allocate_scratch(size_for_shape(&[rows, children], DataType::F32))
            .map_err(WeaverEncodeError::Backend)?;
        self.top_children.encode(
            &candidate_logits,
            step.candidate_scores,
            step.candidate_ids,
            &mut token_ids,
            &mut model_logprobs,
            rows as u32,
            candidates as u32,
            children as u32,
            encoder,
        );
        Ok((token_ids, model_logprobs))
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
        current: Allocation<B>,
        prefix_qkv: &Allocation<B>,
        state_qkv: &mut Allocation<B>,
        node_capacity: u32,
        step: &WeaverStepBatch<'_, B>,
        prefix_length: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let rows = step.row_count;
        let normalized = self.pre_attention_norm.encode(&current, 0, rows, None, encoder)?;
        let current_qkv = self.qkv_projection.encode(normalized, rows, encoder)?;
        let metadata_row_bytes = rows * DataType::U32.size_in_bytes();
        let ancestor_counts = (step.node_metadata, MetadataIdx::AncestorCount as usize * metadata_row_bytes);
        let node_indices = (step.node_metadata, MetadataIdx::TreeSlot as usize * metadata_row_bytes);
        let mut attention = encoder.allocate_scratch(size_for_shape(&[rows, self.model_dim], DATA_TYPE))?;
        self.last_query_attention.encode(
            prefix_qkv,
            &*state_qkv,
            &current_qkv,
            step.ancestor_indices,
            ancestor_counts,
            &mut attention,
            rows as u32,
            prefix_length as u32,
            step.ancestor_stride as u32,
            node_capacity,
            self.attention_scale,
            encoder,
        );
        // Separate dispatch so the node arena is read-only above; the encoder
        // serializes it after the attention that reads the ancestor slots.
        self.node_cache_write.encode(
            &current_qkv,
            state_qkv,
            node_indices,
            self.model_dim as u32,
            node_capacity,
            (rows * 2 * self.model_dim) as u32,
            encoder,
        );
        attention = self.out_projection.encode(attention, rows, encoder)?;
        let mut residual = current;
        self.residual_add.encode(&mut residual, &mut attention, (rows * self.model_dim) as u32, encoder);
        let normalized = self.pre_mlp_norm.encode(&residual, 0, rows, None, encoder)?;
        let mut mlp = self.up_projection.encode(normalized, rows, encoder)?;
        self.activation.encode(
            None::<&Allocation<B>>,
            &mut mlp,
            (rows * self.hidden_dim) as u32,
            crate::backends::common::gpu_types::ActivationType::GELUExact,
            encoder,
        );
        let mut output = self.down_projection.encode(mlp, rows, encoder)?;
        self.residual_add.encode(&mut output, &mut residual, (rows * self.model_dim) as u32, encoder);
        Ok(residual)
    }

    fn encode_prefix(
        &self,
        hidden: Allocation<B>,
        rows: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Allocation<B>), B::Error> {
        let normalized = self.pre_attention_norm.encode(&hidden, 0, rows, None, encoder)?;
        let qkv = self.qkv_projection.encode(normalized, rows, encoder)?;
        let mut queries =
            encoder.allocate_scratch(size_for_shape(&[self.num_heads, rows, self.head_dim], DATA_TYPE))?;
        let mut keys = encoder.allocate_scratch(size_for_shape(&[rows, self.model_dim], DATA_TYPE))?;
        let mut values = encoder.allocate_scratch(size_for_shape(&[rows, self.model_dim], DATA_TYPE))?;
        self.attention_prepare.encode(
            &qkv,
            &mut queries,
            Some(&mut keys),
            Some(&mut values),
            None::<&Allocation<B>>,
            None::<&Allocation<B>>,
            self.num_heads as u32,
            Some(self.num_heads as u32),
            self.head_dim as u32,
            None,
            Some(0),
            rows as u32,
            encoder,
        );
        let state_type = AttentionStateType::Full {
            length: 0,
        };
        let attention = self.prefix_attention.encode(
            AttentionCoreEncodeArguments {
                queries: &queries,
                keys: &keys,
                values: &values,
                suffix_length: rows,
                trie: None,
                sinks: None,
                state_type: &state_type,
            },
            encoder,
        )?;
        let mut attention = self.out_projection.encode(attention, rows, encoder)?;
        let mut residual = hidden;
        self.residual_add.encode(&mut residual, &mut attention, (rows * self.model_dim) as u32, encoder);
        let normalized = self.pre_mlp_norm.encode(&residual, 0, rows, None, encoder)?;
        let mut mlp = self.up_projection.encode(normalized, rows, encoder)?;
        self.activation.encode(
            None::<&Allocation<B>>,
            &mut mlp,
            (rows * self.hidden_dim) as u32,
            crate::backends::common::gpu_types::ActivationType::GELUExact,
            encoder,
        );
        let mut output = self.down_projection.encode(mlp, rows, encoder)?;
        self.residual_add.encode(&mut output, &mut residual, (rows * self.model_dim) as u32, encoder);
        Ok((residual, qkv))
    }
}
