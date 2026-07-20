#![allow(dead_code)]

use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, AllocationType, Backend, Context, Encoder, Kernels,
        gpu_types::trie::TrieNode,
        kernel::{TensorAddSwapKernel, radix_top_k_small::RadixTopKSmall},
    },
    config::{
        dflash::{DFlashDraftConfig, DFlashDraftLayerConfig},
        mlp::AnyMLPConfig,
        normalization::NormalizationConfig,
        rope::AnyRoPEConfig,
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError},
        mixer::{
            Mixer, MixerState,
            attention::{Attention, AttentionNewError, AttentionState, rope::PrecalculatedRoPE},
        },
        mlp::{Mlp, MlpBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar},
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

const RADIX_TOP_K_MAX: usize = 512;

pub struct DFlashState<B: Backend> {
    layer_states: Box<[AttentionState<B>]>,
    context_length: usize,
    context_capacity: usize,
    last_target_hidden: Option<Allocation<B>>,
}

impl<B: Backend> DFlashState<B> {
    pub(crate) fn last_target_hidden(&self) -> Option<&Allocation<B>> {
        self.last_target_hidden.as_ref()
    }
}

pub(crate) struct DFlashDraft<B: Backend> {
    context_projection: Box<dyn Linear<B>>,
    context_norm: Normalization<B>,
    layers: Box<[DFlashDraftLayer<B>]>,
    output_norm: Normalization<B>,
    top_k: <B::Kernels as Kernels>::RadixTopKSmall,
    rope_config: AnyRoPEConfig,
    model_dim: usize,
    max_context_length: usize,
    block_size: usize,
    target_feature_width: usize,
    data_type: DataType,
}

struct DFlashDraftLayer<B: Backend> {
    attention: Attention<B>,
    input_norm: Normalization<B>,
    post_attention_norm: Normalization<B>,
    mlp: Box<dyn Mlp<B>>,
    residual_add: <B::Kernels as Kernels>::TensorAddSwapKernel,
}

#[derive(Debug, Error)]
pub enum DFlashDraftNewError<B: Backend> {
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("MLP error: {0}")]
    Mlp(#[from] MlpBlockError<B>),
    #[error("Attention error: {0}")]
    Attention(#[from] AttentionNewError<B>),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("invalid DFlash attention config: {0}")]
    InvalidAttentionConfig(&'static str),
}

fn plain_norm<B: Backend>(
    context: &B::Context,
    model_dim: usize,
    config: &NormalizationConfig,
    parameter_tree: &ParameterTree<B>,
    data_type: DataType,
) -> Result<Normalization<B>, NormalizationNewError<B>> {
    Normalization::new(model_dim, None, false, false, PostLayerScalar::None, data_type, config, parameter_tree, context)
}

impl<B: Backend> DFlashDraft<B> {
    pub(crate) fn new(
        context: &B::Context,
        config: &DFlashDraftConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, DFlashDraftNewError<B>> {
        if config.layer_configs.is_empty() {
            return Err(DFlashDraftNewError::InvalidAttentionConfig("at least one DFlash layer is required"));
        }

        let context_projection = <dyn Linear<B>>::new(
            config.model_dim * config.target_layer_ids.len(),
            [config.model_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("context_projection")?,
        )?;
        let context_norm = plain_norm(
            context,
            config.model_dim,
            &config.context_norm_config,
            &parameter_tree.subtree("context_norm")?,
            data_type,
        )?;
        let layers_tree = parameter_tree.subtree("layers")?;
        let layers = config
            .layer_configs
            .iter()
            .enumerate()
            .map(|(index, layer_config)| {
                DFlashDraftLayer::new(
                    context,
                    config.model_dim,
                    config.hidden_dim,
                    layer_config,
                    &config.rope_config,
                    &layers_tree.subtree(&index.to_string())?,
                    data_type,
                )
            })
            .collect::<Result<Box<[_]>, DFlashDraftNewError<B>>>()?;
        let output_norm = plain_norm(
            context,
            config.model_dim,
            &config.output_norm_config,
            &parameter_tree.subtree("output_norm")?,
            data_type,
        )?;
        let top_k = <B::Kernels as Kernels>::RadixTopKSmall::new(context, config.vocab_size as u32)
            .map_err(DFlashDraftNewError::Backend)?;

        Ok(Self {
            context_projection,
            context_norm,
            layers,
            output_norm,
            top_k,
            rope_config: config.rope_config.clone(),
            model_dim: config.model_dim,
            max_context_length: *config.rope_config.max_sequence_length(),
            block_size: config.block_size,
            target_feature_width: config.model_dim * config.target_layer_ids.len(),
            data_type,
        })
    }

    pub(crate) fn empty_state(
        &self,
        context_capacity: usize,
        context: &B::Context,
    ) -> Result<DFlashState<B>, B::Error> {
        assert!(context_capacity <= self.max_context_length, "DFlash state capacity exceeds configured RoPE capacity");
        let layer_states = self
            .layers
            .iter()
            .map(|layer| AttentionState::create_empty(&layer.attention, Some(context_capacity), context))
            .collect::<Result<Box<[_]>, B::Error>>()?;
        Ok(DFlashState {
            layer_states,
            context_length: 0,
            context_capacity,
            last_target_hidden: None,
        })
    }

    pub(crate) fn append_state(
        &self,
        state: &mut DFlashState<B>,
        target_features: &[Allocation<B>],
        num_tokens: usize,
        last_target_hidden: &Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        if num_tokens == 0 {
            return Ok(());
        }

        let expected_feature_count = self.target_feature_width / self.model_dim;
        assert_eq!(target_features.len(), expected_feature_count);
        let row_size = self.model_dim * self.data_type.size_in_bytes();
        let layer_feature_size = num_tokens * self.model_dim * self.data_type.size_in_bytes();
        assert!(target_features.iter().all(|features| features.size() == layer_feature_size));
        assert_eq!(last_target_hidden.size(), row_size);
        let context_length = state.context_length + num_tokens;
        assert!(context_length <= state.context_capacity, "DFlash state capacity exceeded");
        assert!(context_length <= self.max_context_length, "DFlash context exceeds configured RoPE capacity");

        let mut projection_input =
            encoder.allocate_scratch(num_tokens * self.target_feature_width * self.data_type.size_in_bytes())?;
        let layer_feature_bytes = self.model_dim * self.data_type.size_in_bytes();
        for (layer_index, features) in target_features.iter().enumerate() {
            for token_index in 0..num_tokens {
                let source_start = token_index * layer_feature_bytes;
                let destination_start = (token_index * target_features.len() + layer_index) * layer_feature_bytes;
                encoder.encode_copy(
                    features,
                    source_start..source_start + layer_feature_bytes,
                    &mut projection_input,
                    destination_start..destination_start + layer_feature_bytes,
                );
            }
        }
        let mut stored_hidden = encoder.context().create_allocation(row_size, AllocationType::Global)?;
        encoder.encode_copy(last_target_hidden, .., &mut stored_hidden, ..);
        state.last_target_hidden = Some(stored_hidden);

        let projected = self.context_projection.encode(projection_input, num_tokens, encoder)?;
        let projected = self.context_norm.encode(&projected, 0, num_tokens, None, encoder)?;
        let token_positions = (state.context_length..state.context_length + num_tokens).collect::<Box<[_]>>();
        let rope = PrecalculatedRoPE::precalculate(&self.rope_config, &token_positions, encoder)?;

        let layer_count = self.layers.len();
        let mut projected = Some(projected);
        for (index, (layer, state_layer)) in self.layers.iter().zip(state.layer_states.iter_mut()).enumerate() {
            state_layer.prepare(state.context_length, num_tokens, encoder.context())?;
            let layer_input = if index + 1 == layer_count {
                projected.take().expect("projected available for last layer")
            } else {
                let source = projected.as_ref().expect("projected available");
                let mut layer_input = encoder.allocate_scratch(source.size())?;
                encoder.encode_copy(source, .., &mut layer_input, ..);
                layer_input
            };
            layer.attention.append_kv(layer_input, Some(&rope), num_tokens, state_layer, encoder)?;
        }

        state.context_length += num_tokens;
        Ok(())
    }

    pub(crate) fn encode_block(
        &self,
        state: &mut DFlashState<B>,
        token_embeddings: Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let batch_dim = self.block_size;
        let nodes = (0..batch_dim)
            .map(|index| TrieNode {
                trie_start: index as u32,
                trie_end: index as u32 + 1,
                height: index as u32,
            })
            .collect::<Box<[_]>>();
        let batch_topology = BatchTopology::new(&nodes, true);
        let token_positions = (state.context_length..state.context_length + batch_dim).collect::<Box<[_]>>();
        let rope = PrecalculatedRoPE::precalculate(&self.rope_config, &token_positions, encoder)?;

        let mut hidden = token_embeddings;
        for (layer, state_layer) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            state_layer.prepare(state.context_length, batch_dim, encoder.context())?;
            let normalized = layer.input_norm.encode(&hidden, 0, batch_dim, None, encoder)?;
            let mut attention_output = layer.attention.encode(
                normalized,
                Some(&rope),
                &batch_topology,
                Some(MaybeMut::Mut(state_layer)),
                encoder,
            )?;
            let mut attention_residual = encoder.allocate_scratch(hidden.size())?;
            encoder.encode_copy(&hidden, .., &mut attention_residual, ..);
            layer.residual_add.encode(
                &mut attention_residual,
                &mut attention_output,
                (batch_dim * self.model_dim) as u32,
                encoder,
            );
            let normalized = layer.post_attention_norm.encode(&attention_residual, 0, batch_dim, None, encoder)?;
            let mut mlp_output = layer.mlp.encode(normalized, batch_dim, encoder)?;

            let mut output = encoder.allocate_scratch(hidden.size())?;
            encoder.encode_copy(&attention_residual, .., &mut output, ..);
            layer.residual_add.encode(&mut output, &mut mlp_output, (batch_dim * self.model_dim) as u32, encoder);
            hidden = output;
        }

        self.output_norm.encode(&hidden, 0, batch_dim, None, encoder)
    }

    pub(crate) fn encode_top_k(
        &self,
        logits: &Allocation<B>,
        rows: usize,
        k: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(Allocation<B>, Allocation<B>), B::Error> {
        assert!(k > 0 && k <= RADIX_TOP_K_MAX);
        let mut ids = encoder.allocate_scratch(size_for_shape(&[rows, k], DataType::U32))?;
        let mut scores = encoder.allocate_scratch(size_for_shape(&[rows, k], DataType::F32))?;
        self.top_k.encode(logits, &mut ids, &mut scores, rows as u32, k as u32, encoder)?;
        Ok((ids, scores))
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }
}

impl<B: Backend> DFlashDraftLayer<B> {
    fn new(
        context: &B::Context,
        model_dim: usize,
        hidden_dim: usize,
        config: &DFlashDraftLayerConfig,
        rope_config: &AnyRoPEConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, DFlashDraftNewError<B>> {
        if config.attention_config.is_causal {
            return Err(DFlashDraftNewError::InvalidAttentionConfig("DFlash attention must be non-causal"));
        }
        let (attention, _) = Attention::new(
            model_dim,
            data_type,
            Some(rope_config),
            &config.attention_config,
            &parameter_tree.subtree("attention")?,
            context,
        )?;
        let input_norm = plain_norm(
            context,
            model_dim,
            &config.input_norm_config,
            &parameter_tree.subtree("input_norm")?,
            data_type,
        )?;
        let post_attention_norm = plain_norm(
            context,
            model_dim,
            &config.post_attention_norm_config,
            &parameter_tree.subtree("post_attention_norm")?,
            data_type,
        )?;
        let mlp_config = AnyMLPConfig::DenseMLPConfig(config.mlp_config.clone());
        let (mlp, _) =
            <dyn Mlp<B>>::new(&mlp_config, model_dim, hidden_dim, context, &parameter_tree.subtree("mlp")?, data_type)?;
        let residual_add = <B::Kernels as Kernels>::TensorAddSwapKernel::new(context, data_type)
            .map_err(DFlashDraftNewError::Backend)?;

        Ok(Self {
            attention,
            input_norm,
            post_attention_norm,
            mlp,
            residual_add,
        })
    }
}
