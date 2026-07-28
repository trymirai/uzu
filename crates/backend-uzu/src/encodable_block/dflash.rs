use thiserror::Error;

use crate::{
    array::size_for_shape,
    backends::common::{
        Allocation, Backend, Encoder, Kernels, gpu_types::trie::TrieNode, kernel::radix_top_k_small::RadixTopKSmall,
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
        embedding::{Embedding, EmbeddingError},
        linear::{Linear, LinearBlockError},
        mixer::{
            Mixer, MixerState,
            attention::{
                ATTENTION_SUFFIX_CAPACITY, Attention, AttentionNewError, AttentionState, rope::PrecalculatedRoPE,
            },
        },
        mlp::{Mlp, MlpBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

const RADIX_TOP_K_MAX: usize = 512;

pub struct DFlashState<B: Backend> {
    layer_states: Box<[AttentionState<B>]>,
    context_length: usize,
    context_capacity: usize,
}

impl<B: Backend> DFlashState<B> {
    pub(crate) fn context_length(&self) -> usize {
        self.context_length
    }
}

pub struct DFlashDraft<B: Backend> {
    target_feature_projection: Box<dyn Linear<B>>,
    projected_feature_norm: Normalization<B>,
    layers: Box<[DFlashDraftLayer<B>]>,
    output_norm: Normalization<B>,
    top_k: <B::Kernels as Kernels>::RadixTopKSmall,
    rope_config: AnyRoPEConfig,
    model_dim: usize,
    max_context_length: usize,
    block_size: usize,
    mask_token_id: u32,
    target_feature_input_dim: usize,
    data_type: DataType,
}

pub struct DFlashDraftOutput<B: Backend> {
    pub candidates: TopKCandidates<B>,
    pub draft_hidden: Allocation<B>,
}

pub struct TopKCandidates<B: Backend> {
    pub ids: Allocation<B>,
    pub scores: Allocation<B>,
    pub rows: usize,
    pub candidates_per_row: usize,
}

struct DFlashDraftLayer<B: Backend> {
    attention: Attention<B>,
    pre_attention_norm: Normalization<B>,
    pre_mlp_norm: Normalization<B>,
    mlp: Box<dyn Mlp<B>>,
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

#[derive(Debug, Error)]
pub enum DFlashDraftEncodeError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError<B>),
}

impl<B: Backend> DFlashDraft<B> {
    pub(crate) fn new(
        context: &B::Context,
        config: &DFlashDraftConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, DFlashDraftNewError<B>> {
        let vocab_size = config.vocab_size as u32;
        let mask_token_id = config.mask_token_id as u32;
        if config.layer_configs.is_empty() {
            return Err(DFlashDraftNewError::InvalidAttentionConfig("at least one DFlash layer is required"));
        }
        if config.block_size == 0 || config.block_size > ATTENTION_SUFFIX_CAPACITY {
            return Err(DFlashDraftNewError::InvalidAttentionConfig(
                "DFlash block_size must be in 1..=attention suffix capacity",
            ));
        }

        let target_feature_projection = <dyn Linear<B>>::new(
            config.model_dim * config.target_layer_ids.len(),
            [config.model_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("context_projection")?,
        )?;
        let projected_feature_norm = Normalization::new(
            config.model_dim,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            data_type,
            &config.context_norm_config,
            &parameter_tree.subtree("context_norm")?,
            context,
        )?;
        let layers_tree = parameter_tree.subtree("layers")?;
        let layers = config
            .layer_configs
            .iter()
            .enumerate()
            .map(|(index, layer_config)| {
                DFlashDraftLayer::new(
                    context,
                    index,
                    config.model_dim,
                    config.hidden_dim,
                    layer_config,
                    &config.rope_config,
                    &layers_tree.subtree(&index.to_string())?,
                    data_type,
                )
            })
            .collect::<Result<Box<[_]>, DFlashDraftNewError<B>>>()?;
        let output_norm = Normalization::new(
            config.model_dim,
            None,
            ShortcutMode::Add,
            PostLayerScalar::None,
            data_type,
            &config.output_norm_config,
            &parameter_tree.subtree("output_norm")?,
            context,
        )?;
        let top_k =
            <B::Kernels as Kernels>::RadixTopKSmall::new(context, vocab_size).map_err(DFlashDraftNewError::Backend)?;

        Ok(Self {
            target_feature_projection,
            projected_feature_norm,
            layers,
            output_norm,
            top_k,
            rope_config: config.rope_config.clone(),
            model_dim: config.model_dim,
            max_context_length: *config.rope_config.max_sequence_length(),
            block_size: config.block_size,
            mask_token_id,
            target_feature_input_dim: config.model_dim * config.target_layer_ids.len(),
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
        })
    }

    pub(crate) fn append_state(
        &self,
        state: &mut DFlashState<B>,
        target_features: &[Allocation<B>],
        accepted_indices: &[usize],
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        if accepted_indices.is_empty() {
            return Ok(());
        }

        let num_tokens = accepted_indices.len();
        let captured_layer_count = self.target_feature_input_dim / self.model_dim;
        assert_eq!(target_features.len(), captured_layer_count);
        let layer_feature_bytes = self.model_dim * self.data_type.size_in_bytes();
        assert!(target_features.iter().all(|features| features.size() % layer_feature_bytes == 0));
        let context_length = state.context_length + num_tokens;
        assert!(context_length <= state.context_capacity, "DFlash state capacity exceeded");
        assert!(context_length <= self.max_context_length, "DFlash context exceeds configured RoPE capacity");

        let mut packed_target_features =
            encoder.allocate_scratch(num_tokens * self.target_feature_input_dim * self.data_type.size_in_bytes())?;
        for (layer_index, features) in target_features.iter().enumerate() {
            for (token_index, &accepted_index) in accepted_indices.iter().enumerate() {
                let source_start = accepted_index * layer_feature_bytes;
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

        let dflash_layer_count = self.layers.len();
        let mut normalized_features = Some(normalized_features);
        for (layer_index, (layer, attention_state)) in self.layers.iter().zip(state.layer_states.iter_mut()).enumerate()
        {
            attention_state.prepare(state.context_length, num_tokens, encoder.context())?;
            let kv_input = if layer_index + 1 == dflash_layer_count {
                normalized_features.take().expect("normalized features available for last layer")
            } else {
                let source = normalized_features.as_ref().expect("normalized features available");
                let mut kv_input = encoder.allocate_scratch(source.size())?;
                encoder.encode_copy(source, .., &mut kv_input, ..);
                kv_input
            };
            layer.attention.append_kv(kv_input, Some(&rope), num_tokens, attention_state, encoder)?;
        }

        state.context_length += num_tokens;
        Ok(())
    }

    fn encode_hidden(
        &self,
        state: &mut DFlashState<B>,
        token_embeddings: Allocation<B>,
        encoder: &mut Encoder<B>,
    ) -> Result<Allocation<B>, B::Error> {
        let batch_dim = self.block_size;
        assert!(
            state.context_length + batch_dim <= self.max_context_length,
            "DFlash block positions exceed configured RoPE capacity"
        );
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
        let mut residual = encoder.allocate_scratch(hidden.size())?;
        for (layer, attention_state) in self.layers.iter().zip(state.layer_states.iter_mut()) {
            attention_state.prepare(state.context_length, batch_dim, encoder.context())?;
            let normalized = layer.pre_attention_norm.encode(&hidden, 0, batch_dim, Some(&mut residual), encoder)?;
            let attention_output = layer.attention.encode(
                normalized,
                Some(&rope),
                &batch_topology,
                Some(MaybeMut::Mut(attention_state)),
                encoder,
            )?;
            let normalized =
                layer.pre_mlp_norm.encode(&attention_output, 0, batch_dim, Some(&mut residual), encoder)?;
            hidden = layer.mlp.encode(normalized, batch_dim, encoder)?;
        }

        self.output_norm.encode(&hidden, 0, batch_dim, Some(&mut residual), encoder)
    }

    pub(crate) fn encode_draft(
        &self,
        state: &mut DFlashState<B>,
        target_output_token: u32,
        target_embedding: &Embedding<B>,
        candidate_count: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<DFlashDraftOutput<B>, DFlashDraftEncodeError<B>> {
        let mut token_ids = encoder
            .allocate_constant(self.block_size * DataType::U32.size_in_bytes())
            .map_err(DFlashDraftEncodeError::Backend)?;
        let mut tokens = vec![self.mask_token_id; self.block_size];
        tokens[0] = target_output_token;
        token_ids.copyin(&tokens);

        let token_embeddings = target_embedding.encode_lookup(&token_ids, self.block_size, encoder)?;
        let draft_hidden =
            self.encode_hidden(state, token_embeddings, encoder).map_err(DFlashDraftEncodeError::Backend)?;

        // The first block row is the target's output token; only the lookahead rows are ranked.
        let lookahead_rows = self.block_size - 1;
        let row_bytes = target_embedding.model_dim() * DataType::BF16.size_in_bytes();
        let mut lookahead_hidden =
            encoder.allocate_scratch(lookahead_rows * row_bytes).map_err(DFlashDraftEncodeError::Backend)?;
        encoder.encode_copy(&draft_hidden, row_bytes..self.block_size * row_bytes, &mut lookahead_hidden, ..);
        let logits = target_embedding.encode_readout(lookahead_rows, &lookahead_hidden, DataType::F32, encoder)?;
        let candidates = self
            .encode_candidates(&logits, lookahead_rows, candidate_count, encoder)
            .map_err(DFlashDraftEncodeError::Backend)?;
        drop(logits);
        drop(token_ids);
        drop(lookahead_hidden);

        Ok(DFlashDraftOutput {
            candidates,
            draft_hidden,
        })
    }

    fn encode_candidates(
        &self,
        logits: &Allocation<B>,
        rows: usize,
        k: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<TopKCandidates<B>, B::Error> {
        assert!(k > 0 && k <= RADIX_TOP_K_MAX);
        let mut ids = encoder.allocate_scratch(size_for_shape(&[rows, k], DataType::U32))?;
        let mut scores = encoder.allocate_scratch(size_for_shape(&[rows, k], DataType::F32))?;
        self.top_k.encode(logits, &mut ids, &mut scores, rows as u32, k as u32, encoder)?;
        Ok(TopKCandidates {
            ids,
            scores,
            rows,
            candidates_per_row: k,
        })
    }
}

impl<B: Backend> DFlashDraftLayer<B> {
    fn new(
        context: &B::Context,
        layer_index: usize,
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
        if config.attention_config.is_kv_sharing {
            return Err(DFlashDraftNewError::InvalidAttentionConfig("DFlash attention must not use KV sharing"));
        }
        let (attention, _) = Attention::new(
            model_dim,
            data_type,
            Some(rope_config),
            &config.attention_config,
            &parameter_tree.subtree("attention")?,
            context,
        )?;
        let pre_attention_norm = Normalization::new(
            model_dim,
            None,
            if layer_index == 0 {
                ShortcutMode::Copy
            } else {
                ShortcutMode::Add
            },
            PostLayerScalar::None,
            data_type,
            &config.input_norm_config,
            &parameter_tree.subtree("input_norm")?,
            context,
        )?;
        let pre_mlp_norm = Normalization::new(
            model_dim,
            None,
            ShortcutMode::Add,
            PostLayerScalar::None,
            data_type,
            &config.post_attention_norm_config,
            &parameter_tree.subtree("post_attention_norm")?,
            context,
        )?;
        let mlp_config = AnyMLPConfig::DenseMLPConfig(config.mlp_config.clone());
        let (mlp, _) =
            <dyn Mlp<B>>::new(&mlp_config, model_dim, hidden_dim, context, &parameter_tree.subtree("mlp")?, data_type)?;
        Ok(Self {
            attention,
            pre_attention_norm,
            pre_mlp_norm,
            mlp,
        })
    }
}
