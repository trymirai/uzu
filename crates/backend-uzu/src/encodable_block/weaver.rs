use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder, Kernels,
        kernel::{
            ActivationKernel, AttentionLastQueryKernel, AttentionPrepareKernel, TensorAddBiasKernel,
            TensorAddSwapKernel, WeaverTopChildrenKernel,
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

pub(crate) struct WeaverPrefix<B: Backend> {
    layer_qkv: Box<[Allocation<B>]>,
    pub(crate) length: usize,
}

pub(crate) struct WeaverBatchStepOutput {
    pub children: Vec<(u32, f32)>,
}

pub(crate) struct WeaverBatchStepInput<'a> {
    pub node_index: usize,
    pub parent_token: u32,
    pub candidates: &'a [u32],
    pub candidate_scores: &'a [f32],
    pub ancestors: &'a [usize],
    pub depth: usize,
}

pub(crate) struct WeaverNodeState<B: Backend> {
    layer_qkv: Box<[Allocation<B>]>,
    capacity: usize,
}

struct EncodedWeaverBatch<B: Backend> {
    child_ids: Allocation<B>,
    child_logprobs: Allocation<B>,
    _retained: Vec<Allocation<B>>,
}

pub(crate) struct Weaver<B: Backend> {
    embedding_norm: WeaverNorm<B>,
    hidden_state_norm: WeaverNorm<B>,
    embedding_projection: Box<dyn Linear<B>>,
    blocks: Box<[WeaverBlock<B>]>,
    output_norm: WeaverNorm<B>,
    hidden_state_projection: Box<dyn Linear<B>>,
    query_projection: Box<dyn Linear<B>>,
    position_embeddings: Allocation<B>,
    position_add: <B::Kernels as Kernels>::TensorAddBiasKernel,
    top_children: <B::Kernels as Kernels>::WeaverTopChildrenKernel,
    model_dim: usize,
    target_model_dim: usize,
    max_depth: usize,
    data_type: DataType,
}

struct WeaverNorm<B: Backend> {
    normalization: Normalization<B>,
    biases: Option<Allocation<B>>,
    bias_kernel: Option<<B::Kernels as Kernels>::TensorAddBiasKernel>,
    dimension: usize,
}

struct WeaverBlock<B: Backend> {
    qkv_projection: Box<dyn Linear<B>>,
    out_projection: Box<dyn Linear<B>>,
    prefix_attention: AttentionCores<B>,
    attention_prepare: <B::Kernels as Kernels>::AttentionPrepareKernel,
    pre_attention_norm: WeaverNorm<B>,
    pre_mlp_norm: WeaverNorm<B>,
    up_projection: Box<dyn Linear<B>>,
    down_projection: Box<dyn Linear<B>>,
    activation: <B::Kernels as Kernels>::ActivationKernel,
    residual_add: <B::Kernels as Kernels>::TensorAddSwapKernel,
    last_query_attention: <B::Kernels as Kernels>::AttentionLastQueryKernel,
    attention_scale: f32,
    model_dim: usize,
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
    data_type: DataType,
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
}

#[derive(Debug, Error)]
pub enum WeaverEncodeError<B: Backend> {
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
}

fn linear<B: Backend>(
    context: &B::Context,
    parameter_tree: &ParameterTree<B>,
    name: &str,
    input_dim: usize,
    output_dim: usize,
    has_biases: bool,
    data_type: DataType,
) -> Result<Box<dyn Linear<B>>, WeaverNewError<B>> {
    Ok(<dyn Linear<B>>::new(input_dim, [output_dim], has_biases, context, data_type, &parameter_tree.subtree(name)?)?)
}

fn plain_norm<B: Backend>(
    context: &B::Context,
    dim: usize,
    config: &NormalizationConfig,
    parameter_tree: &ParameterTree<B>,
    data_type: DataType,
) -> Result<WeaverNorm<B>, WeaverNewError<B>> {
    let mut normalization_config = config.clone();
    normalization_config.has_biases = false;
    let normalization = Normalization::new(
        dim,
        None,
        false,
        false,
        PostLayerScalar::None,
        data_type,
        &normalization_config,
        parameter_tree,
        context,
    )?;
    let biases = config
        .has_biases
        .then(|| parameter_tree.leaf("biases")?.validate(&[dim], DataType::F32)?.read_allocation())
        .transpose()?;
    let bias_kernel = biases
        .as_ref()
        .map(|_| <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, data_type, DataType::F32, true))
        .transpose()
        .map_err(WeaverNewError::Backend)?;
    Ok(WeaverNorm {
        normalization,
        biases,
        bias_kernel,
        dimension: dim,
    })
}

impl<B: Backend> WeaverNorm<B> {
    fn encode(
        &self,
        input: &Allocation<B>,
        row_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let mut output = self.normalization.encode(input, 0, row_count, None, encoder)?;
        if let (Some(biases), Some(kernel)) = (&self.biases, &self.bias_kernel) {
            kernel.encode(
                None::<&Allocation<B>>,
                biases,
                &mut output,
                self.dimension as u32,
                (row_count * self.dimension) as u32,
                encoder,
            );
        }
        Ok(output)
    }
}

impl<B: Backend> Weaver<B> {
    pub(crate) fn create_node_state(
        &self,
        capacity: usize,
        context: &B::Context,
    ) -> Result<WeaverNodeState<B>, WeaverEncodeError<B>> {
        assert!(capacity > 0);
        let qkv_size = size_for_shape(&[capacity, 3, self.model_dim], self.data_type);
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
        data_type: DataType,
    ) -> Result<Self, WeaverNewError<B>> {
        if config.num_heads == 0 || !config.model_dim.is_multiple_of(config.num_heads) {
            return Err(WeaverNewError::InvalidHeadConfig);
        }
        let embedding_norm = plain_norm(
            context,
            config.target_embedding_dim,
            &config.norm_config,
            &parameter_tree.subtree("embedding_norm")?,
            data_type,
        )?;
        let hidden_state_norm = plain_norm(
            context,
            config.target_model_dim,
            &config.norm_config,
            &parameter_tree.subtree("hidden_state_norm")?,
            data_type,
        )?;
        let embedding_projection = linear(
            context,
            parameter_tree,
            "embedding_projection",
            config.target_embedding_dim,
            config.model_dim,
            true,
            data_type,
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
                    data_type,
                )
            })
            .collect::<Result<Box<[_]>, WeaverNewError<B>>>()?;
        let output_norm = plain_norm(
            context,
            config.model_dim,
            &config.norm_config,
            &parameter_tree.subtree("output_norm")?,
            data_type,
        )?;
        let hidden_state_projection = linear(
            context,
            parameter_tree,
            "hidden_state_projection",
            config.target_model_dim,
            config.model_dim,
            true,
            data_type,
        )?;
        let query_projection = linear(
            context,
            parameter_tree,
            "query_projection",
            config.model_dim,
            config.target_model_dim,
            false,
            data_type,
        )?;
        let position_embeddings = parameter_tree
            .leaf("position_embeddings")?
            .validate(&[config.max_depth, config.model_dim], DataType::F32)?
            .read_allocation()?;
        let position_add = <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, data_type, DataType::F32, true)
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
            top_children,
            model_dim: config.model_dim,
            target_model_dim: config.target_model_dim,
            max_depth: config.max_depth,
            data_type,
        })
    }

    pub(crate) fn prompt_prefix(
        &self,
        target_hidden: &Allocation<B>,
        lookaheads: &Allocation<B>,
        lookahead_offset: usize,
        lookahead_count: usize,
        context: &B::Context,
    ) -> Result<WeaverPrefix<B>, WeaverEncodeError<B>> {
        let length = lookahead_count + 1;
        assert!(lookahead_count <= self.max_depth);
        let row_bytes = self.target_model_dim * self.data_type.size_in_bytes();
        assert!(target_hidden.size() >= row_bytes);
        assert!(lookaheads.size() >= (lookahead_offset + lookahead_count) * row_bytes);

        let mut encoder = Encoder::new(context).map_err(WeaverEncodeError::Backend)?;
        let mut input = encoder
            .allocate_scratch(size_for_shape(&[length, self.target_model_dim], self.data_type))
            .map_err(WeaverEncodeError::Backend)?;
        encoder.encode_copy(target_hidden, 0..row_bytes, &mut input, 0..row_bytes);
        encoder.encode_copy(
            lookaheads,
            lookahead_offset * row_bytes..(lookahead_offset + lookahead_count) * row_bytes,
            &mut input,
            row_bytes..length * row_bytes,
        );
        let normalized =
            self.hidden_state_norm.encode(&input, length, &mut encoder).map_err(WeaverEncodeError::Backend)?;
        let mut hidden = self
            .hidden_state_projection
            .encode(normalized, length, &mut encoder)
            .map_err(WeaverEncodeError::Backend)?;
        self.add_positions(&mut hidden, lookahead_count, 0, 1, &mut encoder);

        let mut layer_qkv = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (next_hidden, qkv) =
                block.encode_prefix(hidden, length, &mut encoder).map_err(WeaverEncodeError::Backend)?;
            let mut cached_qkv =
                context.create_allocation(qkv.size(), AllocationType::Global).map_err(WeaverEncodeError::Backend)?;
            encoder.encode_copy(&qkv, .., &mut cached_qkv, ..);
            layer_qkv.push(cached_qkv);
            hidden = next_hidden;
        }
        let completed = encoder.end_encoding().submit().wait_until_completed().map_err(WeaverEncodeError::Backend)?;
        drop(input);
        drop(hidden);
        drop(completed);
        Ok(WeaverPrefix {
            layer_qkv: layer_qkv.into_boxed_slice(),
            length,
        })
    }

    pub(crate) fn step_batch(
        &self,
        prefix: &WeaverPrefix<B>,
        inputs: &[WeaverBatchStepInput<'_>],
        state: &mut WeaverNodeState<B>,
        children: usize,
        target_embedding: &Embedding<B>,
        context: &B::Context,
    ) -> Result<Vec<WeaverBatchStepOutput>, WeaverEncodeError<B>> {
        assert!(!inputs.is_empty());
        let mut encoder = Encoder::new(context).map_err(WeaverEncodeError::Backend)?;
        let encoded = self.encode_step_batch(prefix, inputs, state, children, target_embedding, &mut encoder)?;
        let completed = encoder.end_encoding().submit().wait_until_completed().map_err(WeaverEncodeError::Backend)?;
        let child_ids = encoded.child_ids.copyout::<u32>();
        let child_logprobs = encoded.child_logprobs.copyout::<f32>();
        let outputs = inputs
            .iter()
            .enumerate()
            .map(|(batch_index, _)| WeaverBatchStepOutput {
                children: (0..children)
                    .map(|child| {
                        let index = batch_index * children + child;
                        (child_ids[index], child_logprobs[index])
                    })
                    .collect(),
            })
            .collect();
        drop(encoded);
        drop(completed);
        Ok(outputs)
    }

    fn encode_step_batch(
        &self,
        prefix: &WeaverPrefix<B>,
        inputs: &[WeaverBatchStepInput<'_>],
        state: &mut WeaverNodeState<B>,
        children: usize,
        target_embedding: &Embedding<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<EncodedWeaverBatch<B>, WeaverEncodeError<B>> {
        let rows = inputs.len();
        let candidates = inputs[0].candidates.len();
        assert!(candidates > 0);
        assert!(children > 0 && children <= candidates);
        assert!(inputs.iter().all(|input| {
            input.candidates.len() == candidates
                && input.candidate_scores.len() == candidates
                && input.depth < self.max_depth
                && input.node_index < state.capacity
        }));

        let mut token_ids =
            encoder.allocate_constant(size_for_shape(&[rows], DataType::U64)).map_err(WeaverEncodeError::Backend)?;
        token_ids.copyin(&inputs.iter().map(|input| input.parent_token as u64).collect::<Vec<_>>());
        let mut candidate_ids = encoder
            .allocate_constant(size_for_shape(&[rows, candidates], DataType::U64))
            .map_err(WeaverEncodeError::Backend)?;
        candidate_ids.copyin(
            &inputs.iter().flat_map(|input| input.candidates.iter().map(|&token| token as u64)).collect::<Vec<_>>(),
        );
        let mut candidate_scores = encoder
            .allocate_constant(size_for_shape(&[rows, candidates], DataType::F32))
            .map_err(WeaverEncodeError::Backend)?;
        candidate_scores
            .copyin(&inputs.iter().flat_map(|input| input.candidate_scores.iter().copied()).collect::<Vec<_>>());
        let token_embedding = target_embedding.encode_lookup(&token_ids, rows, encoder)?;
        let embedding_normalized =
            self.embedding_norm.encode(&token_embedding, rows, encoder).map_err(WeaverEncodeError::Backend)?;
        let mut current = self
            .embedding_projection
            .encode(embedding_normalized, rows, encoder)
            .map_err(WeaverEncodeError::Backend)?;
        for (row, input) in inputs.iter().enumerate() {
            self.add_positions(&mut current, 1, input.depth, row, encoder);
        }

        let ancestor_stride = inputs.iter().map(|input| input.ancestors.len()).max().unwrap().max(1);
        let mut ancestor_indices = encoder
            .allocate_constant(size_for_shape(&[rows, ancestor_stride], DataType::U32))
            .map_err(WeaverEncodeError::Backend)?;
        let mut ancestors = vec![0u32; rows * ancestor_stride];
        for (row, input) in inputs.iter().enumerate() {
            for (offset, &ancestor) in input.ancestors.iter().enumerate() {
                ancestors[row * ancestor_stride + offset] = ancestor as u32;
            }
        }
        ancestor_indices.copyin(&ancestors);
        let mut ancestor_counts =
            encoder.allocate_constant(size_for_shape(&[rows], DataType::U32)).map_err(WeaverEncodeError::Backend)?;
        ancestor_counts.copyin(&inputs.iter().map(|input| input.ancestors.len() as u32).collect::<Vec<_>>());
        let mut node_indices =
            encoder.allocate_constant(size_for_shape(&[rows], DataType::U32)).map_err(WeaverEncodeError::Backend)?;
        node_indices.copyin(&inputs.iter().map(|input| input.node_index as u32).collect::<Vec<_>>());

        let mut retained = vec![ancestor_indices, ancestor_counts, node_indices];
        for (layer_index, block) in self.blocks.iter().enumerate() {
            let normalized =
                block.pre_attention_norm.encode(&current, rows, encoder).map_err(WeaverEncodeError::Backend)?;
            let current_qkv =
                block.qkv_projection.encode(normalized, rows, encoder).map_err(WeaverEncodeError::Backend)?;
            let mut attention_rows = encoder
                .allocate_scratch(size_for_shape(&[rows, self.model_dim], self.data_type))
                .map_err(WeaverEncodeError::Backend)?;
            block.last_query_attention.encode(
                &prefix.layer_qkv[layer_index],
                &mut state.layer_qkv[layer_index],
                &current_qkv,
                &retained[0],
                &retained[1],
                &retained[2],
                &mut attention_rows,
                rows as u32,
                prefix.length as u32,
                ancestor_stride as u32,
                block.attention_scale,
                encoder,
            );
            attention_rows =
                block.out_projection.encode(attention_rows, rows, encoder).map_err(WeaverEncodeError::Backend)?;
            retained.push(current_qkv);
            let mut residual = encoder.allocate_scratch(current.size()).map_err(WeaverEncodeError::Backend)?;
            encoder.encode_copy(&current, .., &mut residual, ..);
            block.residual_add.encode(&mut residual, &mut attention_rows, (rows * self.model_dim) as u32, encoder);
            let normalized = block.pre_mlp_norm.encode(&residual, rows, encoder).map_err(WeaverEncodeError::Backend)?;
            let mut mlp = block.up_projection.encode(normalized, rows, encoder).map_err(WeaverEncodeError::Backend)?;
            block.activation.encode(
                None::<&Allocation<B>>,
                &mut mlp,
                (rows * block.hidden_dim) as u32,
                crate::backends::common::gpu_types::ActivationType::GELUExact,
                encoder,
            );
            let mut output = block.down_projection.encode(mlp, rows, encoder).map_err(WeaverEncodeError::Backend)?;
            block.residual_add.encode(&mut output, &mut residual, (rows * self.model_dim) as u32, encoder);
            current = residual;
        }

        let output_normalized = self.output_norm.encode(&current, rows, encoder).map_err(WeaverEncodeError::Backend)?;
        let query =
            self.query_projection.encode(output_normalized, rows, encoder).map_err(WeaverEncodeError::Backend)?;
        let candidate_logits =
            target_embedding.encode_readout_sparse(&query, &candidate_ids, rows, candidates, encoder)?;
        let mut child_ids = encoder
            .allocate_scratch(size_for_shape(&[rows, children], DataType::U32))
            .map_err(WeaverEncodeError::Backend)?;
        let mut child_logprobs = encoder
            .allocate_scratch(size_for_shape(&[rows, children], DataType::F32))
            .map_err(WeaverEncodeError::Backend)?;
        self.top_children.encode(
            &candidate_logits,
            &candidate_scores,
            &candidate_ids,
            &mut child_ids,
            &mut child_logprobs,
            rows as u32,
            candidates as u32,
            children as u32,
            encoder,
        );
        retained.extend([
            token_ids,
            candidate_ids,
            token_embedding,
            current,
            candidate_scores,
            candidate_logits,
            query,
        ]);
        Ok(EncodedWeaverBatch {
            child_ids,
            child_logprobs,
            _retained: retained,
        })
    }

    fn add_positions(
        &self,
        hidden: &mut Allocation<B>,
        rows: usize,
        start_depth: usize,
        row_offset: usize,
        encoder: &mut Encoder<B>,
    ) {
        let row_bytes = self.model_dim * self.data_type.size_in_bytes();
        let position_bytes = self.model_dim * DataType::F32.size_in_bytes();
        for row in 0..rows {
            let depth = start_depth + row;
            assert!(depth < self.max_depth);
            self.position_add.encode(
                None::<&Allocation<B>>,
                (&self.position_embeddings, depth * position_bytes),
                (&mut *hidden, (row + row_offset) * row_bytes),
                self.model_dim as u32,
                self.model_dim as u32,
                encoder,
            );
        }
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
        data_type: DataType,
    ) -> Result<Self, WeaverNewError<B>> {
        let head_dim = model_dim / num_heads;
        let attention_scale = 1.0 / (head_dim as f32).sqrt();
        let qkv_projection =
            linear(context, parameter_tree, "qkv_projection", model_dim, 3 * model_dim, false, data_type)?;
        let out_projection = linear(context, parameter_tree, "out_projection", model_dim, model_dim, false, data_type)?;
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
                data_type,
            },
            context,
        )
        .map_err(WeaverNewError::Backend)?;
        let attention_prepare =
            <B::Kernels as Kernels>::AttentionPrepareKernel::new(context, data_type, DataType::F32, true, false)
                .map_err(WeaverNewError::Backend)?;
        let pre_attention_norm =
            plain_norm(context, model_dim, norm_config, &parameter_tree.subtree("pre_attention_norm")?, data_type)?;
        let pre_mlp_norm =
            plain_norm(context, model_dim, norm_config, &parameter_tree.subtree("pre_mlp_norm")?, data_type)?;
        let up_projection = linear(context, parameter_tree, "up_projection", model_dim, hidden_dim, true, data_type)?;
        let down_projection =
            linear(context, parameter_tree, "down_projection", hidden_dim, model_dim, true, data_type)?;
        let activation = <B::Kernels as Kernels>::ActivationKernel::new(context, data_type, true)
            .map_err(WeaverNewError::Backend)?;
        let residual_add =
            <B::Kernels as Kernels>::TensorAddSwapKernel::new(context, data_type).map_err(WeaverNewError::Backend)?;
        let last_query_attention =
            <B::Kernels as Kernels>::AttentionLastQueryKernel::new(context, head_dim as u32, num_heads as u32)
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
            attention_scale,
            model_dim,
            hidden_dim,
            num_heads,
            head_dim,
            data_type,
        })
    }

    fn encode_prefix(
        &self,
        hidden: Allocation<B>,
        rows: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Allocation<B>), B::Error> {
        let normalized = self.pre_attention_norm.encode(&hidden, rows, encoder)?;
        let qkv = self.qkv_projection.encode(normalized, rows, encoder)?;
        let mut queries =
            encoder.allocate_scratch(size_for_shape(&[self.num_heads, rows, self.head_dim], self.data_type))?;
        let mut keys = encoder.allocate_scratch(size_for_shape(&[rows, self.model_dim], self.data_type))?;
        let mut values = encoder.allocate_scratch(size_for_shape(&[rows, self.model_dim], self.data_type))?;
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
        let mut residual = encoder.allocate_scratch(hidden.size())?;
        encoder.encode_copy(&hidden, .., &mut residual, ..);
        self.residual_add.encode(&mut residual, &mut attention, (rows * self.model_dim) as u32, encoder);
        let normalized = self.pre_mlp_norm.encode(&residual, rows, encoder)?;
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
