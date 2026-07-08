use thiserror::Error;

use crate::{
    backends::common::Backend,
    config::{dflash::DFlashDraftConfig, mlp::AnyMLPConfig},
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError},
        mlp::{Mlp, MlpBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

#[allow(dead_code)]
pub(crate) struct DFlashDraft<B: Backend> {
    _context_projection: Box<dyn Linear<B>>,
    _context_norm: Normalization<B>,
    _layers: Box<[DFlashDraftLayer<B>]>,
    _output_norm: Normalization<B>,
    model_dim: usize,
    max_context_length: Option<usize>,
}

#[allow(dead_code)]
struct DFlashDraftLayer<B: Backend> {
    _attention: DFlashAttention<B>,
    _input_norm: Normalization<B>,
    _post_attention_norm: Normalization<B>,
    _mlp: Box<dyn Mlp<B>>,
}

#[allow(dead_code)]
struct DFlashAttention<B: Backend> {
    _query_projection: Box<dyn Linear<B>>,
    _key_value_projection: Box<dyn Linear<B>>,
    _output_projection: Box<dyn Linear<B>>,
    _query_norm: Normalization<B>,
    _key_norm: Normalization<B>,
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
}

#[allow(dead_code)]
impl<B: Backend> DFlashDraft<B> {
    pub(crate) fn new(
        context: &B::Context,
        config: &DFlashDraftConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, DFlashDraftNewError<B>> {
        let context_projection = <dyn Linear<B>>::new(
            config.model_dim * config.target_layer_ids.len(),
            [config.model_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("context_projection")?,
        )?;
        let context_norm = Normalization::new(
            config.model_dim,
            None,
            false,
            false,
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
                    config.model_dim,
                    config.hidden_dim,
                    layer_config,
                    &layers_tree.subtree(&index.to_string())?,
                    data_type,
                )
            })
            .collect::<Result<Box<[_]>, DFlashDraftNewError<B>>>()?;
        let output_norm = Normalization::new(
            config.model_dim,
            None,
            false,
            false,
            PostLayerScalar::None,
            data_type,
            &config.output_norm_config,
            &parameter_tree.subtree("output_norm")?,
            context,
        )?;
        let max_context_length = config
            .layer_configs
            .iter()
            .map(|layer_config| *layer_config.attention_config.rope_config.max_sequence_length())
            .min();

        Ok(Self {
            _context_projection: context_projection,
            _context_norm: context_norm,
            _layers: layers,
            _output_norm: output_norm,
            model_dim: config.model_dim,
            max_context_length,
        })
    }

    pub(crate) fn model_dim(&self) -> usize {
        self.model_dim
    }

    pub(crate) fn max_context_length(&self) -> Option<usize> {
        self.max_context_length
    }
}

impl<B: Backend> DFlashDraftLayer<B> {
    fn new(
        context: &B::Context,
        model_dim: usize,
        hidden_dim: usize,
        config: &crate::config::dflash::DFlashDraftLayerConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, DFlashDraftNewError<B>> {
        let attention = DFlashAttention::new(
            context,
            model_dim,
            &config.attention_config,
            &parameter_tree.subtree("attention")?,
            data_type,
        )?;
        let input_norm = Normalization::new(
            model_dim,
            None,
            false,
            false,
            PostLayerScalar::None,
            data_type,
            &config.input_norm_config,
            &parameter_tree.subtree("input_norm")?,
            context,
        )?;
        let post_attention_norm = Normalization::new(
            model_dim,
            None,
            false,
            false,
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
            _attention: attention,
            _input_norm: input_norm,
            _post_attention_norm: post_attention_norm,
            _mlp: mlp,
        })
    }
}

impl<B: Backend> DFlashAttention<B> {
    fn new(
        context: &B::Context,
        model_dim: usize,
        config: &crate::config::dflash::DFlashAttentionConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, DFlashDraftNewError<B>> {
        let query_dim = config.num_heads * config.head_dim;
        let key_value_dim = config.num_key_value_heads * config.head_dim;
        let query_projection = <dyn Linear<B>>::new(
            model_dim,
            [query_dim],
            config.has_attention_biases,
            context,
            data_type,
            &parameter_tree.subtree("query_projection")?,
        )?;
        let key_value_projection = <dyn Linear<B>>::new(
            model_dim,
            [2 * key_value_dim],
            config.has_attention_biases,
            context,
            data_type,
            &parameter_tree.subtree("key_value_projection")?,
        )?;
        let output_projection = <dyn Linear<B>>::new(
            query_dim,
            [model_dim],
            config.has_output_biases,
            context,
            data_type,
            &parameter_tree.subtree("output_projection")?,
        )?;
        let query_norm = Normalization::new(
            config.head_dim,
            None,
            false,
            false,
            PostLayerScalar::None,
            data_type,
            &config.query_norm_config,
            &parameter_tree.subtree("query_norm")?,
            context,
        )?;
        let key_norm = Normalization::new(
            config.head_dim,
            None,
            false,
            false,
            PostLayerScalar::None,
            data_type,
            &config.key_norm_config,
            &parameter_tree.subtree("key_norm")?,
            context,
        )?;

        Ok(Self {
            _query_projection: query_projection,
            _key_value_projection: key_value_projection,
            _output_projection: output_projection,
            _query_norm: query_norm,
            _key_norm: key_norm,
        })
    }
}
