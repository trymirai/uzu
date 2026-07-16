use half::bf16;
use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels,
        gpu_types::trie::TrieNode,
        kernel::{ActivationKernel, TensorAddBiasKernel, TensorAddSwapKernel},
    },
    config::{normalization::NormalizationConfig, token_mixer::attention::AttentionConfig, weaver::WeaverConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding::{Embedding, EmbeddingError},
        linear::{Linear, LinearBlockError},
        mixer::attention::{Attention, AttentionNewError},
        mlp::MlpBlockError,
        normalization::{Normalization, NormalizationNewError, PostLayerScalar},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

/// Host-side prefix state used by the correctness-first Weaver implementation.
/// Each entry is the input to one Weaver block for every prefix position.
pub(crate) struct WeaverPrefix {
    pub(crate) layer_inputs: Box<[Vec<bf16>]>,
    pub(crate) length: usize,
}

/// Host-side state for one expanded Weaver node.
pub(crate) struct WeaverNode {
    pub(crate) layer_inputs: Box<[Vec<bf16>]>,
}

pub(crate) struct WeaverStepOutput {
    pub logits: Vec<f32>,
    pub node: WeaverNode,
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
    attention: Attention<B>,
    pre_attention_norm: WeaverNorm<B>,
    pre_mlp_norm: WeaverNorm<B>,
    up_projection: Box<dyn Linear<B>>,
    down_projection: Box<dyn Linear<B>>,
    activation: <B::Kernels as Kernels>::ActivationKernel,
    residual_add: <B::Kernels as Kernels>::TensorAddSwapKernel,
    model_dim: usize,
    hidden_dim: usize,
}

#[derive(Debug, Error)]
pub enum WeaverNewError<B: Backend> {
    #[error("parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("MLP error: {0}")]
    Mlp(#[from] MlpBlockError<B>),
    #[error("attention error: {0}")]
    Attention(#[from] AttentionNewError<B>),
    #[error("backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("attention config error: {0}")]
    Config(#[from] serde_json::Error),
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
        let embedding_projection = <dyn Linear<B>>::new(
            config.target_embedding_dim,
            [config.model_dim],
            true,
            context,
            data_type,
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
        let hidden_state_projection = <dyn Linear<B>>::new(
            config.target_model_dim,
            [config.model_dim],
            true,
            context,
            data_type,
            &parameter_tree.subtree("hidden_state_projection")?,
        )?;
        let query_projection = <dyn Linear<B>>::new(
            config.model_dim,
            [config.target_model_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("query_projection")?,
        )?;
        let position_embeddings = parameter_tree
            .leaf("position_embeddings")?
            .validate(&[config.max_depth, config.model_dim], DataType::F32)?
            .read_allocation()?;
        assert_eq!(
            position_embeddings.size(),
            config.max_depth * config.model_dim * DataType::F32.size_in_bytes(),
            "Weaver position embedding allocation does not match max_depth"
        );
        let position_add = <B::Kernels as Kernels>::TensorAddBiasKernel::new(context, data_type, DataType::F32, true)
            .map_err(WeaverNewError::Backend)?;
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
            model_dim: config.model_dim,
            target_model_dim: config.target_model_dim,
            max_depth: config.max_depth,
            data_type,
        })
    }

    pub(crate) fn prompt_prefix(
        &self,
        target_hidden: &[bf16],
        lookaheads: &[bf16],
        context: &B::Context,
    ) -> Result<WeaverPrefix, WeaverEncodeError<B>> {
        assert_eq!(target_hidden.len(), self.target_model_dim);
        assert_eq!(lookaheads.len() % self.target_model_dim, 0);
        let lookahead_count = lookaheads.len() / self.target_model_dim;
        let length = lookahead_count + 1;
        assert!(lookahead_count <= self.max_depth);

        let mut encoder = Encoder::new(context).map_err(WeaverEncodeError::Backend)?;
        let mut input = encoder
            .allocate_constant(size_for_shape(&[length, self.target_model_dim], self.data_type))
            .map_err(WeaverEncodeError::Backend)?;
        let mut host_input = Vec::with_capacity(length * self.target_model_dim);
        host_input.extend_from_slice(target_hidden);
        host_input.extend_from_slice(lookaheads);
        input.copyin(&host_input);
        let normalized =
            self.hidden_state_norm.encode(&input, length, &mut encoder).map_err(WeaverEncodeError::Backend)?;
        let mut hidden = self
            .hidden_state_projection
            .encode(normalized, length, &mut encoder)
            .map_err(WeaverEncodeError::Backend)?;
        self.add_positions(&mut hidden, lookahead_count, 0, 1, &mut encoder);

        let nodes = (0..length)
            .map(|index| TrieNode {
                trie_start: index as u32,
                trie_end: index as u32 + 1,
                height: index as u32,
            })
            .collect::<Box<[_]>>();
        let topology = BatchTopology::new(&nodes, true);
        let mut layer_inputs = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let mut snapshot = encoder.allocate_constant(hidden.size()).map_err(WeaverEncodeError::Backend)?;
            encoder.encode_copy(&hidden, .., &mut snapshot, ..);
            layer_inputs.push(snapshot);
            hidden = block.forward(hidden, &topology, length, &mut encoder).map_err(WeaverEncodeError::Backend)?;
        }
        let completed = encoder.end_encoding().submit().wait_until_completed().map_err(WeaverEncodeError::Backend)?;
        drop(input);
        let layer_input_allocations = layer_inputs;
        let layer_inputs =
            layer_input_allocations.iter().map(|allocation| allocation.copyout::<bf16>()).collect::<Box<[_]>>();
        drop(hidden);
        drop(layer_input_allocations);
        drop(completed);
        Ok(WeaverPrefix {
            layer_inputs,
            length,
        })
    }

    pub(crate) fn step(
        &self,
        prefix: &WeaverPrefix,
        parent_token: u32,
        candidates: &[u32],
        candidate_scores: &[f32],
        ancestors: &[&WeaverNode],
        depth: usize,
        target_embedding: &Embedding<B>,
        context: &B::Context,
    ) -> Result<WeaverStepOutput, WeaverEncodeError<B>> {
        assert_eq!(candidates.len(), candidate_scores.len());
        assert!(!candidates.is_empty());
        assert!(depth < self.max_depth);
        let mut encoder = Encoder::new(context).map_err(WeaverEncodeError::Backend)?;
        let mut token_ids =
            encoder.allocate_constant(size_for_shape(&[1], DataType::U64)).map_err(WeaverEncodeError::Backend)?;
        token_ids.copyin(&[parent_token as u64]);
        let mut candidate_ids = encoder
            .allocate_constant(size_for_shape(&[candidates.len()], DataType::U64))
            .map_err(WeaverEncodeError::Backend)?;
        candidate_ids.copyin(&candidates.iter().map(|&token| token as u64).collect::<Box<[_]>>());
        let token_embedding = target_embedding.encode_lookup(&token_ids, 1, &mut encoder)?;
        let embedding_normalized =
            self.embedding_norm.encode(&token_embedding, 1, &mut encoder).map_err(WeaverEncodeError::Backend)?;
        let mut current = self
            .embedding_projection
            .encode(embedding_normalized, 1, &mut encoder)
            .map_err(WeaverEncodeError::Backend)?;
        self.add_positions(&mut current, 1, depth, 0, &mut encoder);

        let mut node_inputs = Vec::with_capacity(self.blocks.len());
        for (layer_index, block) in self.blocks.iter().enumerate() {
            let mut snapshot = encoder.allocate_constant(current.size()).map_err(WeaverEncodeError::Backend)?;
            encoder.encode_copy(&current, .., &mut snapshot, ..);
            let mut sequence = Vec::with_capacity(prefix.length + ancestors.len() + 1);
            sequence.extend_from_slice(&prefix.layer_inputs[layer_index]);
            for ancestor in ancestors {
                sequence.extend_from_slice(&ancestor.layer_inputs[layer_index]);
            }
            let sequence_length = prefix.length + ancestors.len() + 1;
            sequence.resize(sequence_length * self.model_dim, bf16::from_f32(0.0));
            let mut sequence_allocation = encoder
                .allocate_constant(size_for_shape(&[sequence_length, self.model_dim], self.data_type))
                .map_err(WeaverEncodeError::Backend)?;
            sequence_allocation.copyin(&sequence);
            let row_size = self.model_dim * self.data_type.size_in_bytes();
            let current_start = (sequence_length - 1) * row_size;
            encoder.encode_copy(&current, .., &mut sequence_allocation, current_start..sequence_length * row_size);
            let topology_nodes = (0..sequence_length)
                .map(|index| TrieNode {
                    trie_start: index as u32,
                    trie_end: index as u32 + 1,
                    height: index as u32,
                })
                .collect::<Box<[_]>>();
            let topology = BatchTopology::new(&topology_nodes, true);
            let normalized_sequence = block
                .pre_attention_norm
                .encode(&sequence_allocation, sequence_length, &mut encoder)
                .map_err(WeaverEncodeError::Backend)?;
            let attention = block
                .attention
                .encode_stateless(normalized_sequence, &topology, &mut encoder)
                .map_err(WeaverEncodeError::Backend)?;
            let mut attention_current = encoder
                .allocate_scratch(size_for_shape(&[1, self.model_dim], self.data_type))
                .map_err(WeaverEncodeError::Backend)?;
            encoder.encode_copy(
                &attention,
                (sequence_length - 1) * row_size..sequence_length * row_size,
                &mut attention_current,
                ..,
            );
            let mut attention_residual =
                encoder.allocate_scratch(current.size()).map_err(WeaverEncodeError::Backend)?;
            encoder.encode_copy(&current, .., &mut attention_residual, ..);
            block.residual_add.encode(
                &mut attention_residual,
                &mut attention_current,
                self.model_dim as u32,
                &mut encoder,
            );
            let normalized =
                block.pre_mlp_norm.encode(&attention_residual, 1, &mut encoder).map_err(WeaverEncodeError::Backend)?;
            let mut mlp =
                block.up_projection.encode(normalized, 1, &mut encoder).map_err(WeaverEncodeError::Backend)?;
            block.activation.encode(
                None::<&Allocation<B>>,
                &mut mlp,
                block.hidden_dim as u32,
                crate::backends::common::gpu_types::ActivationType::GELUExact,
                &mut encoder,
            );
            let mut output = block.down_projection.encode(mlp, 1, &mut encoder).map_err(WeaverEncodeError::Backend)?;
            block.residual_add.encode(&mut output, &mut attention_residual, self.model_dim as u32, &mut encoder);
            current = attention_residual;
            node_inputs.push(snapshot);
        }

        let output_normalized =
            self.output_norm.encode(&current, 1, &mut encoder).map_err(WeaverEncodeError::Backend)?;
        let query =
            self.query_projection.encode(output_normalized, 1, &mut encoder).map_err(WeaverEncodeError::Backend)?;
        let candidate_logits =
            target_embedding.encode_readout_candidates(&query, &candidate_ids, 1, candidates.len(), &mut encoder)?;
        let completed = encoder.end_encoding().submit().wait_until_completed().map_err(WeaverEncodeError::Backend)?;
        let logits = candidate_logits.copyout::<f32>();
        let node_input_allocations = node_inputs;
        let node_inputs =
            node_input_allocations.iter().map(|allocation| allocation.copyout::<bf16>()).collect::<Box<[_]>>();
        drop(candidate_logits);
        drop(query);
        drop(current);
        drop(node_input_allocations);
        drop(token_embedding);
        drop(candidate_ids);
        drop(token_ids);
        drop(completed);
        let logits = candidates
            .iter()
            .zip(candidate_scores)
            .enumerate()
            .map(|(index, (_, &score))| score + logits[index])
            .collect();
        Ok(WeaverStepOutput {
            logits,
            node: WeaverNode {
                layer_inputs: node_inputs,
            },
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
        let attention_config: AttentionConfig = serde_json::from_value(serde_json::json!({
            "type": "AttentionConfig",
            "qkv_projection_config": {},
            "out_projection_config": {},
            "query_norm_config": null,
            "key_norm_config": null,
            "num_heads": num_heads,
            "num_groups": num_heads,
            "head_dim": head_dim,
            "is_causal": true,
            "scale": 1.0 / (head_dim as f32).sqrt(),
            "sliding_window_size": null,
            "logit_soft_cap": null,
            "has_sinks": false,
            "has_qkv_biases": false,
            "has_out_biases": false,
            "gate_projection_config": null,
            "normalize_values": false,
            "is_kv_sharing": false,
        }))?;
        let (attention, _) = Attention::new(model_dim, data_type, None, &attention_config, parameter_tree, context)?;
        let pre_attention_norm =
            plain_norm(context, model_dim, norm_config, &parameter_tree.subtree("pre_attention_norm")?, data_type)?;
        let pre_mlp_norm =
            plain_norm(context, model_dim, norm_config, &parameter_tree.subtree("pre_mlp_norm")?, data_type)?;
        let up_projection = <dyn Linear<B>>::new(
            model_dim,
            [hidden_dim],
            true,
            context,
            data_type,
            &parameter_tree.subtree("up_projection")?,
        )?;
        let down_projection = <dyn Linear<B>>::new(
            hidden_dim,
            [model_dim],
            true,
            context,
            data_type,
            &parameter_tree.subtree("down_projection")?,
        )?;
        let activation = <B::Kernels as Kernels>::ActivationKernel::new(context, data_type, true)
            .map_err(WeaverNewError::Backend)?;
        let residual_add =
            <B::Kernels as Kernels>::TensorAddSwapKernel::new(context, data_type).map_err(WeaverNewError::Backend)?;
        Ok(Self {
            attention,
            pre_attention_norm,
            pre_mlp_norm,
            up_projection,
            down_projection,
            activation,
            residual_add,
            model_dim,
            hidden_dim,
        })
    }

    fn forward(
        &self,
        hidden: Allocation<B>,
        topology: &BatchTopology,
        rows: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let normalized = self.pre_attention_norm.encode(&hidden, rows, encoder)?;
        let mut attention = self.attention.encode_stateless(normalized, topology, encoder)?;
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
        Ok(residual)
    }
}
