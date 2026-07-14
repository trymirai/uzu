use thiserror::Error;

use crate::{
    backends::common::Backend,
    config::{
        dflash::{DFlashAttentionConfig, DFlashDraftConfig, DFlashDraftLayerConfig},
        mlp::AnyMLPConfig,
        normalization::NormalizationConfig,
    },
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError},
        mlp::{Mlp, MlpBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar},
    },
    parameters::{ParameterLoaderError, ParameterTree},
};

// TODO: remove once traversal verification consumes the loaded DFlash model.
#[allow(dead_code)]
pub(crate) struct DFlashDraft<B: Backend> {
    context_projection: Box<dyn Linear<B>>,
    context_norm: Normalization<B>,
    layers: Box<[DFlashDraftLayer<B>]>,
    output_norm: Normalization<B>,
    model_dim: usize,
    max_context_length: Option<usize>,
}

// TODO: remove once traversal verification consumes the loaded DFlash model.
#[allow(dead_code)]
struct DFlashDraftLayer<B: Backend> {
    attention: DFlashAttention<B>,
    input_norm: Normalization<B>,
    post_attention_norm: Normalization<B>,
    mlp: Box<dyn Mlp<B>>,
}

// TODO: remove once traversal verification consumes the loaded DFlash model.
#[allow(dead_code)]
struct DFlashAttention<B: Backend> {
    query_projection: Box<dyn Linear<B>>,
    key_value_projection: Box<dyn Linear<B>>,
    output_projection: Box<dyn Linear<B>>,
    query_norm: Normalization<B>,
    key_norm: Normalization<B>,
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
        let max_context_length = config
            .layer_configs
            .iter()
            .map(|layer_config| *layer_config.attention_config.rope_config.max_sequence_length())
            .min();
        Ok(Self {
            context_projection,
            context_norm,
            layers,
            output_norm,
            model_dim: config.model_dim,
            max_context_length,
        })
    }
}

impl<B: Backend> DFlashDraftLayer<B> {
    fn new(
        context: &B::Context,
        model_dim: usize,
        hidden_dim: usize,
        config: &DFlashDraftLayerConfig,
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

        Ok(Self {
            attention,
            input_norm,
            post_attention_norm,
            mlp,
        })
    }
}

impl<B: Backend> DFlashAttention<B> {
    fn new(
        context: &B::Context,
        model_dim: usize,
        config: &DFlashAttentionConfig,
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
        let query_norm = plain_norm(
            context,
            config.head_dim,
            &config.query_norm_config,
            &parameter_tree.subtree("query_norm")?,
            data_type,
        )?;
        let key_norm = plain_norm(
            context,
            config.head_dim,
            &config.key_norm_config,
            &parameter_tree.subtree("key_norm")?,
            data_type,
        )?;

        Ok(Self {
            query_projection,
            key_value_projection,
            output_projection,
            query_norm,
            key_norm,
        })
    }
}
