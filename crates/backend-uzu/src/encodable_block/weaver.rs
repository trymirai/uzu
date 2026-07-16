use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend},
    config::{normalization::NormalizationConfig, weaver::WeaverConfig},
    data_type::DataType,
    encodable_block::linear::{Linear, LinearBlockError},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[allow(dead_code)]
pub struct Weaver<B: Backend> {
    embedding_norm: WeaverNorm<B>,
    hidden_state_norm: WeaverNorm<B>,
    embedding_projection: Box<dyn Linear<B>>,
    blocks: Box<[WeaverBlock<B>]>,
    output_norm: WeaverNorm<B>,
    hidden_state_projection: Box<dyn Linear<B>>,
    query_projection: Box<dyn Linear<B>>,
    position_embeddings: Allocation<B>,
}

#[allow(dead_code)]
struct WeaverNorm<B: Backend> {
    scales: Allocation<B>,
    biases: Option<Allocation<B>>,
}

#[allow(dead_code)]
struct WeaverBlock<B: Backend> {
    pre_attention_norm: WeaverNorm<B>,
    qkv_projection: Box<dyn Linear<B>>,
    out_projection: Box<dyn Linear<B>>,
    pre_mlp_norm: WeaverNorm<B>,
    up_projection: Box<dyn Linear<B>>,
    down_projection: Box<dyn Linear<B>>,
}

#[derive(Debug, Error)]
pub enum WeaverNewError<B: Backend> {
    #[error("parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("model_dim must be divisible by num_heads")]
    InvalidHeadConfig,
}

fn plain_norm<B: Backend>(
    dim: usize,
    config: &NormalizationConfig,
    parameter_tree: &ParameterTree<B>,
) -> Result<WeaverNorm<B>, ParameterLoaderError<B>> {
    let scales = parameter_tree.leaf("scales")?.validate(&[dim], DataType::F32)?.read_allocation()?;
    let biases = config
        .has_biases
        .then(|| parameter_tree.leaf("biases")?.validate(&[dim], DataType::F32)?.read_allocation())
        .transpose()?;
    Ok(WeaverNorm {
        scales,
        biases,
    })
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

        let embedding_norm =
            plain_norm(config.target_embedding_dim, &config.norm_config, &parameter_tree.subtree("embedding_norm")?)?;
        let hidden_state_norm =
            plain_norm(config.target_model_dim, &config.norm_config, &parameter_tree.subtree("hidden_state_norm")?)?;
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
                    &config.norm_config,
                    &blocks_tree.subtree(&index.to_string())?,
                    data_type,
                )
            })
            .collect::<Result<Box<[_]>, WeaverNewError<B>>>()?;
        let output_norm = plain_norm(config.model_dim, &config.norm_config, &parameter_tree.subtree("output_norm")?)?;
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

        Ok(Self {
            embedding_norm,
            hidden_state_norm,
            embedding_projection,
            blocks,
            output_norm,
            hidden_state_projection,
            query_projection,
            position_embeddings,
        })
    }
}

impl<B: Backend> WeaverBlock<B> {
    fn new(
        context: &B::Context,
        model_dim: usize,
        hidden_dim: usize,
        norm_config: &NormalizationConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, WeaverNewError<B>> {
        let pre_attention_norm = plain_norm(model_dim, norm_config, &parameter_tree.subtree("pre_attention_norm")?)?;
        let qkv_projection = <dyn Linear<B>>::new(
            model_dim,
            [model_dim, model_dim, model_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("qkv_projection")?,
        )?;
        let out_projection = <dyn Linear<B>>::new(
            model_dim,
            [model_dim],
            false,
            context,
            data_type,
            &parameter_tree.subtree("out_projection")?,
        )?;
        let pre_mlp_norm = plain_norm(model_dim, norm_config, &parameter_tree.subtree("pre_mlp_norm")?)?;
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

        Ok(Self {
            pre_attention_norm,
            qkv_projection,
            out_projection,
            pre_mlp_norm,
            up_projection,
            down_projection,
        })
    }
}
