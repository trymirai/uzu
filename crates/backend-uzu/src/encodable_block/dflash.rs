use std::any::Any;

use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{Allocation, Backend, Encoder, gpu_types::trie::TrieNode},
    config::{dflash::DFlashDraftConfig, rope::AnyRoPEConfig, token_mixer::AnyTokenMixerConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding::{Embedding, EmbeddingError},
        linear::{Linear, LinearBlockError},
        mixer::{
            MixerState,
            attention::{
                ATTENTION_SUFFIX_CAPACITY, Attention, AttentionNewError, AttentionState, rope::PrecalculatedRoPE,
            },
        },
        mlp::MlpBlockError,
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        transformer_layer::{TransformerLayer, TransformerLayerError},
    },
    parameters::ParameterTree,
    utils::maybe_mut::MaybeMut,
};

pub struct DFlashState<B: Backend> {
    layer_states: Box<[Box<dyn MixerState<B>>]>,
    context_length: u32,
    context_capacity: u32,
}

impl<B: Backend> DFlashState<B> {
    pub fn context_length(&self) -> u32 {
        self.context_length
    }
}

pub struct DFlash<B: Backend> {
    target_feature_projection: Box<dyn Linear<B>>,
    projected_feature_norm: Normalization<B>,
    state_kv_projection: Box<dyn Linear<B>>,
    layer_kv_dim: u32,
    layers: Box<[TransformerLayer<B>]>,
    output_norm: Normalization<B>,
    rope_config: AnyRoPEConfig,
    model_dim: u32,
    max_context_length: u32,
    block_size: u32,
    mask_token_id: u32,
    target_feature_input_dim: u32,
    data_type: DataType,
}

pub struct DFlashOutput<B: Backend> {
    pub draft_hidden: Allocation<B>,
    pub logits: Allocation<B>,
}

#[derive(Debug, Error)]
pub enum DFlashNewError<B: Backend> {
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("MLP error: {0}")]
    Mlp(#[from] MlpBlockError<B>),
    #[error("Attention error: {0}")]
    Attention(#[from] AttentionNewError<B>),
    #[error("Transformer layer error: {0}")]
    TransformerLayer(#[from] TransformerLayerError<B>),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("invalid DFlash attention config: {0}")]
    InvalidAttentionConfig(&'static str),
}

#[derive(Debug, Error)]
pub enum DFlashEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
}

impl<B: Backend> DFlash<B> {
    pub fn new(
        context: &B::Context,
        config: &DFlashDraftConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, DFlashNewError<B>> {
        let mask_token_id = config.mask_token_id as u32;
        assert!(config.block_size <= ATTENTION_SUFFIX_CAPACITY, "DFlash block_size exceeds attention suffix capacity");
        let target_feature_projection = <dyn Linear<B>>::new(
            config.model_dim * config.target_layer_ids.len() as u32,
            [config.model_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("context_projection"),
        )?;
        let projected_feature_norm = Normalization::new(
            config.model_dim,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            data_type,
            &config.context_norm_config,
            &parameter_tree.subtree("context_norm"),
            context,
        )?;
        let layers_tree = parameter_tree.subtree("layers");
        let first_layer_config = config.layer_configs.first().expect("DFlash draft model must have at least one layer");
        let AnyTokenMixerConfig::AttentionConfig(attention_config) = &first_layer_config.mixer_config else {
            return Err(DFlashNewError::InvalidAttentionConfig("DFlash layers must use attention mixers"));
        };
        let layer_kv_dim = 2 * attention_config.num_groups * attention_config.head_dim;
        let num_layers = config.layer_configs.len() as u32;
        let state_kv_projection = <dyn Linear<B>>::new(
            config.model_dim,
            [num_layers * layer_kv_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("state_kv_projection"),
        )?;
        let layers = config
            .layer_configs
            .iter()
            .zip(0u32..)
            .map(|(layer_config, index)| {
                TransformerLayer::new(
                    context,
                    config.model_dim,
                    config.hidden_dim,
                    num_layers,
                    layer_config,
                    index,
                    &layers_tree.subtree(&index.to_string()),
                    data_type,
                )
            })
            .collect::<Result<Box<[_]>, TransformerLayerError<B>>>()?;
        let output_norm = Normalization::new(
            config.model_dim,
            None,
            ShortcutMode::Add,
            PostLayerScalar::None,
            data_type,
            &config.output_norm_config,
            &parameter_tree.subtree("output_norm"),
            context,
        )?;

        Ok(Self {
            target_feature_projection,
            projected_feature_norm,
            state_kv_projection,
            layer_kv_dim,
            layers,
            output_norm,
            rope_config: config.rope_config.clone(),
            model_dim: config.model_dim,
            max_context_length: *config.rope_config.max_sequence_length(),
            block_size: config.block_size,
            mask_token_id,
            target_feature_input_dim: config.model_dim * config.target_layer_ids.len() as u32,
            data_type,
        })
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    pub fn empty_state(
        &self,
        context_capacity: u32,
        context: &B::Context,
    ) -> Result<DFlashState<B>, B::Error> {
        assert!(context_capacity <= self.max_context_length, "DFlash state capacity exceeds configured RoPE capacity");
        let layer_states = self
            .layers
            .iter()
            .map(|layer| layer.mixer.create_empty_state(Some(context_capacity), context))
            .collect::<Result<Box<[_]>, B::Error>>()?;
        Ok(DFlashState {
            layer_states,
            context_length: 0,
            context_capacity,
        })
    }

    pub fn encode_accept(
        &self,
        state: &mut DFlashState<B>,
        target_features: &[Allocation<B>],
        accepted_indices: &[u32],
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        if accepted_indices.is_empty() {
            return Ok(());
        }

        encoder.push_debug_group("dflash accept");

        let num_tokens = accepted_indices.len() as u32;
        let captured_layer_count = self.target_feature_input_dim / self.model_dim;
        assert_eq!(target_features.len() as u32, captured_layer_count);
        let layer_feature_bytes = size_for_shape(&[self.model_dim], self.data_type);
        assert!(target_features.iter().all(|features| features.size() % layer_feature_bytes == 0));
        let context_length = state.context_length + num_tokens;
        assert!(context_length <= state.context_capacity, "DFlash state capacity exceeded");
        assert!(context_length <= self.max_context_length, "DFlash context exceeds configured RoPE capacity");

        let mut packed_target_features =
            encoder.allocate_scratch_for_shape(&[num_tokens, self.target_feature_input_dim], self.data_type)?;
        for (layer_index, features) in target_features.iter().enumerate() {
            for (token_index, &accepted_index) in accepted_indices.iter().enumerate() {
                let source_start = accepted_index as usize * layer_feature_bytes;
                assert!(source_start + layer_feature_bytes <= features.size(), "accepted index out of feature bounds");
                let destination_start = (token_index * target_features.len() + layer_index) * layer_feature_bytes;
                encoder.encode_copy(
                    features,
                    source_start..source_start + layer_feature_bytes,
                    &mut packed_target_features,
                    destination_start..destination_start + layer_feature_bytes,
                );
            }
        }
        let projected_features = self.target_feature_projection.encode(packed_target_features, num_tokens, encoder)?;
        let normalized_features =
            self.projected_feature_norm.encode(&projected_features, 0, num_tokens, None, encoder)?;
        let token_positions = (state.context_length..state.context_length + num_tokens).collect::<Box<[_]>>();
        let rope = PrecalculatedRoPE::precalculate(&self.rope_config, &token_positions, encoder)?;

        let projected_kv = self.state_kv_projection.encode(normalized_features, num_tokens, encoder)?;
        let layer_kv_bytes = size_for_shape(&[self.layer_kv_dim], self.data_type);
        let kv_chunk = |chunk_index: usize| chunk_index * layer_kv_bytes..(chunk_index + 1) * layer_kv_bytes;
        let mut layer_key_values = (0..self.layers.len())
            .map(|_| encoder.allocate_scratch(num_tokens as usize * layer_kv_bytes))
            .collect::<Result<Box<[_]>, _>>()?;
        for (layer_index, key_value) in layer_key_values.iter_mut().enumerate() {
            for token_index in 0..num_tokens as usize {
                encoder.encode_copy(
                    &projected_kv,
                    kv_chunk(token_index * self.layers.len() + layer_index),
                    key_value,
                    kv_chunk(token_index),
                );
            }
        }
        for ((layer, mixer_state), key_value) in
            self.layers.iter().zip(state.layer_states.iter_mut()).zip(layer_key_values)
        {
            mixer_state.prepare(state.context_length, num_tokens, encoder.context())?;
            let attention = (layer.mixer.as_ref() as &dyn Any)
                .downcast_ref::<Attention<B>>()
                .expect("DFlash draft layers must use attention mixers");
            let attention_state = (mixer_state.as_mut() as &mut dyn Any)
                .downcast_mut::<AttentionState<B>>()
                .expect("DFlash draft layer states must be attention states");
            attention.append_projected_kv(key_value, &rope, num_tokens, attention_state, encoder)?;
        }

        state.context_length += num_tokens;

        encoder.pop_debug_group();

        Ok(())
    }

    pub fn encode_draft(
        &self,
        state: &mut DFlashState<B>,
        target_output_token: u32,
        target_embedding: &Embedding<B>,
        batch_size: u32,
        encoder: &mut Encoder<B>,
    ) -> Result<DFlashOutput<B>, DFlashEncodeError<B>> {
        encoder.push_debug_group("dflash draft");

        assert!(batch_size >= 2 && batch_size <= self.block_size, "batch size exceeds DFlash block size");
        assert!(
            state.context_length + batch_size <= self.max_context_length,
            "DFlash block positions exceed configured RoPE capacity"
        );

        let mut tokens = vec![self.mask_token_id; batch_size as usize];
        tokens[0] = target_output_token;
        let token_ids = encoder.allocate_constant_from_slice(&tokens).map_err(DFlashEncodeError::Backend)?;

        let token_embeddings = target_embedding.encode_lookup(&token_ids, batch_size, encoder)?;

        let nodes = (0..batch_size)
            .map(|index| TrieNode {
                trie_start: index,
                trie_end: batch_size - 1,
                height: index,
            })
            .collect::<Box<[_]>>();
        let batch_topology = BatchTopology::new(&nodes, true);
        let token_positions = (state.context_length..state.context_length + batch_size).collect::<Box<[_]>>();
        let rope = PrecalculatedRoPE::precalculate(&self.rope_config, &token_positions, encoder)
            .map_err(DFlashEncodeError::Backend)?;

        let mut hidden = token_embeddings;
        let mut residual = encoder.allocate_scratch(hidden.size()).map_err(DFlashEncodeError::Backend)?;
        for (layer, mixer_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            mixer_state
                .prepare(state.context_length, batch_size, encoder.context())
                .map_err(DFlashEncodeError::Backend)?;
            hidden = layer
                .encode(
                    hidden,
                    &mut residual,
                    None,
                    Some(&rope),
                    &batch_topology,
                    Some(MaybeMut::Mut(mixer_state.as_mut())),
                    encoder,
                )
                .map_err(DFlashEncodeError::Backend)?;
        }
        let draft_hidden = self
            .output_norm
            .encode(&hidden, 0, batch_size, Some(&mut residual), encoder)
            .map_err(DFlashEncodeError::Backend)?;

        let row_bytes = size_for_shape(&[target_embedding.model_dim()], DataType::BF16);
        let lookahead_rows = row_bytes..batch_size as usize * row_bytes;
        let mut lookahead_hidden =
            encoder.allocate_scratch(lookahead_rows.len()).map_err(DFlashEncodeError::Backend)?;
        encoder.encode_copy(&draft_hidden, lookahead_rows, &mut lookahead_hidden, ..);
        let logits = target_embedding.encode_readout(batch_size - 1, &lookahead_hidden, DataType::F32, encoder)?;

        encoder.pop_debug_group();

        Ok(DFlashOutput {
            draft_hidden,
            logits,
        })
    }
}
