use thiserror::Error;

use crate::{
    backends::common::{Allocation, Backend},
    config::{normalization::NormalizationConfig, weaver::WeaverConfig},
    data_type::DataType,
    encodable_block::linear::{Linear, LinearBlockError},
    parameters::{ParameterLoaderError, ParameterTree},
};

#[allow(dead_code)] // TODO: remove once Weaver traversal wiring consumes the loaded model.
pub(crate) struct Weaver<B: Backend> {
    embed_norm: WeaverNorm<B>,
    output_norm: WeaverNorm<B>,
    token_in: Box<dyn Linear<B>>,
    blocks: Box<[WeaverBlock<B>]>,
    out_norm: WeaverNorm<B>,
    proposal_in: Box<dyn Linear<B>>,
    lm_head_query_in: Box<dyn Linear<B>>,
    pos_emb: Allocation<B>,
}

#[allow(dead_code)] // TODO: remove once Weaver traversal wiring uses norms during forward.
struct WeaverNorm<B: Backend> {
    _scales: Allocation<B>,
    _biases: Option<Allocation<B>>,
}

#[allow(dead_code)] // TODO: remove once Weaver traversal wiring consumes loaded blocks.
struct WeaverBlock<B: Backend> {
    norm_attn: WeaverNorm<B>,
    q_proj: Box<dyn Linear<B>>,
    k_proj: Box<dyn Linear<B>>,
    v_proj: Box<dyn Linear<B>>,
    o_proj: Box<dyn Linear<B>>,
    norm_mlp: WeaverNorm<B>,
    fc1: Box<dyn Linear<B>>,
    fc2: Box<dyn Linear<B>>,
}

#[derive(Debug, Error)]
pub enum WeaverNewError<B: Backend> {
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("invalid Weaver head config")]
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
        _scales: scales,
        _biases: biases,
    })
}

impl<B: Backend> Weaver<B> {
    pub(crate) fn new(
        context: &B::Context,
        config: &WeaverConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, WeaverNewError<B>> {
        if config.num_heads == 0 || config.d_rank % config.num_heads != 0 {
            return Err(WeaverNewError::InvalidHeadConfig);
        }

        let embed_norm = plain_norm(config.d_embed, &config.norm_config, &parameter_tree.subtree("embed_norm")?)?;
        let output_norm = plain_norm(config.d_model, &config.norm_config, &parameter_tree.subtree("output_norm")?)?;
        let token_in = <dyn Linear<B>>::new(
            config.d_embed,
            [config.d_rank],
            true,
            context,
            data_type,
            &parameter_tree.subtree("token_in")?,
        )?;
        let blocks_tree = parameter_tree.subtree("blocks")?;
        let blocks = (0..config.num_layers)
            .map(|index| {
                WeaverBlock::new(
                    context,
                    config.d_rank,
                    config.mlp_dim,
                    &config.norm_config,
                    &blocks_tree.subtree(&index.to_string())?,
                    data_type,
                )
            })
            .collect::<Result<Box<[_]>, WeaverNewError<B>>>()?;
        let out_norm = plain_norm(config.d_rank, &config.norm_config, &parameter_tree.subtree("out_norm")?)?;
        let proposal_in = <dyn Linear<B>>::new(
            config.d_model,
            [config.d_rank],
            true,
            context,
            data_type,
            &parameter_tree.subtree("proposal_in")?,
        )?;
        let lm_head_query_in = <dyn Linear<B>>::new(
            config.d_rank,
            [config.d_model],
            false,
            context,
            data_type,
            &parameter_tree.subtree("lm_head_query_in")?,
        )?;
        let pos_emb =
            parameter_tree.leaf("pos_emb")?.validate(&[config.k, config.d_rank], DataType::F32)?.read_allocation()?;

        Ok(Self {
            embed_norm,
            output_norm,
            token_in,
            blocks,
            out_norm,
            proposal_in,
            lm_head_query_in,
            pos_emb,
        })
    }
}

impl<B: Backend> WeaverBlock<B> {
    fn new(
        context: &B::Context,
        d_rank: usize,
        mlp_dim: usize,
        norm_config: &NormalizationConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, WeaverNewError<B>> {
        let norm_attn = plain_norm(d_rank, norm_config, &parameter_tree.subtree("norm_attn")?)?;
        let q_proj =
            <dyn Linear<B>>::new(d_rank, [d_rank], false, context, data_type, &parameter_tree.subtree("q_proj")?)?;
        let k_proj =
            <dyn Linear<B>>::new(d_rank, [d_rank], false, context, data_type, &parameter_tree.subtree("k_proj")?)?;
        let v_proj =
            <dyn Linear<B>>::new(d_rank, [d_rank], false, context, data_type, &parameter_tree.subtree("v_proj")?)?;
        let o_proj =
            <dyn Linear<B>>::new(d_rank, [d_rank], false, context, data_type, &parameter_tree.subtree("o_proj")?)?;
        let norm_mlp = plain_norm(d_rank, norm_config, &parameter_tree.subtree("norm_mlp")?)?;
        let fc1 = <dyn Linear<B>>::new(d_rank, [mlp_dim], true, context, data_type, &parameter_tree.subtree("fc1")?)?;
        let fc2 = <dyn Linear<B>>::new(mlp_dim, [d_rank], true, context, data_type, &parameter_tree.subtree("fc2")?)?;

        Ok(Self {
            norm_attn,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            norm_mlp,
            fc1,
            fc2,
        })
    }
}
