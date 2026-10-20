use std::sync::OnceLock;

use crate::{
    backends::{
        Backend,
        common::{BufferMut, Context},
    },
    data_type::DataType,
    encodable_block::conf_head::{
        CONF_HEAD_NODE_OUTPUTS, CONF_HEAD_POOL_INPUTS, CONF_HEAD_SLOT_NUMBERS, ConfidenceHeadError,
        config::ConfidenceHeadConfig,
    },
    safetensors::{ParameterLoaderError, ParameterTree},
};

pub struct ConfidenceHeadWeights<B: Backend> {
    pub draft_norm_scales: B::GlobalBuffer,
    pub draft_norm_biases: B::GlobalBuffer,
    pub draft_weights: B::GlobalBuffer,
    pub draft_biases: B::GlobalBuffer,
    pub token_norm_scales: B::GlobalBuffer,
    pub token_norm_biases: B::GlobalBuffer,
    pub token_weights: B::GlobalBuffer,
    pub token_biases: B::GlobalBuffer,
    pub token_table: OnceLock<B::GlobalBuffer>,
}

impl<B: Backend> ConfidenceHeadWeights<B> {
    pub fn new(
        tree: &ParameterTree<B>,
        config: &ConfidenceHeadConfig,
    ) -> Result<Self, ParameterLoaderError<B>> {
        let draft_norm_tree = tree.subtree("draft_norm");
        let draft_norm_scales = draft_norm_tree.read_buffer("scales", &[config.model_width], DataType::F32)?;
        let draft_norm_biases = draft_norm_tree.read_buffer("biases", &[config.model_width], DataType::F32)?;

        let draft_tree = tree.subtree("draft");
        let draft_weights = draft_tree.subtree("weights").read_buffer(
            "weights",
            &[config.hidden, config.model_width],
            DataType::F32,
        )?;
        let draft_biases = draft_tree.read_buffer("biases", &[config.hidden], DataType::F32)?;

        let token_norm_tree = tree.subtree("token_norm");
        let token_norm_scales = token_norm_tree.read_buffer("scales", &[config.model_width], DataType::F32)?;
        let token_norm_biases = token_norm_tree.read_buffer("biases", &[config.model_width], DataType::F32)?;

        let token_tree = tree.subtree("token_projection");
        let token_weights = token_tree.subtree("weights").read_buffer(
            "weights",
            &[config.hidden, config.model_width],
            DataType::F32,
        )?;
        let token_biases = token_tree.read_buffer("biases", &[config.hidden], DataType::F32)?;

        Ok(ConfidenceHeadWeights {
            draft_norm_scales,
            draft_norm_biases,
            draft_weights,
            draft_biases,
            token_norm_scales,
            token_norm_biases,
            token_weights,
            token_biases,
            token_table: OnceLock::new(),
        })
    }
}

pub fn get_local_weights_buffer<B: Backend>(
    context: &B::Context,
    config: &ConfidenceHeadConfig,
    tree: &ParameterTree<B>,
) -> Result<B::GlobalBuffer, ConfidenceHeadError<B>> {
    let local_weights = read_local_weights(config, tree).map_err(ConfidenceHeadError::Parameter)?;
    let mut local_weights_buffer =
        context.create_buffer(local_weights.len() * size_of::<f32>()).map_err(ConfidenceHeadError::Backend)?;
    local_weights_buffer.copyin(&local_weights);
    Ok(local_weights_buffer)
}

fn read_local_weights<B: Backend>(
    config: &ConfidenceHeadConfig,
    tree: &ParameterTree<B>,
) -> Result<Vec<f32>, ParameterLoaderError<B>> {
    let edge_input = 3 * config.hidden + CONF_HEAD_SLOT_NUMBERS as u32;
    let weights = [
        tree.subtree("pool_set.entry.weights")
            .read_slice::<f32>("weights", &[config.pool_hidden, CONF_HEAD_POOL_INPUTS as u32])?,
        tree.subtree("pool_set.entry").read_slice::<f32>("biases", &[config.pool_hidden])?,
        tree.subtree("pool_set.hidden.weights")
            .read_slice::<f32>("weights", &[config.pool_hidden, config.pool_hidden])?,
        tree.subtree("pool_set.hidden").read_slice::<f32>("biases", &[config.pool_hidden])?,
        tree.subtree("pool_set.out.weights").read_slice::<f32>("weights", &[config.hidden, 2 * config.pool_hidden])?,
        tree.subtree("pool_set.out").read_slice::<f32>("biases", &[config.hidden])?,
        tree.subtree("slot").read_slice::<f32>("means", &[CONF_HEAD_SLOT_NUMBERS as u32])?,
        tree.subtree("slot").read_slice::<f32>("stds", &[CONF_HEAD_SLOT_NUMBERS as u32])?,
        tree.subtree("slot").read_slice::<f32>("keep", &[CONF_HEAD_SLOT_NUMBERS as u32])?,
        tree.subtree("edge.input_norm").read_slice::<f32>("scales", &[edge_input])?,
        tree.subtree("edge.input_norm").read_slice::<f32>("biases", &[edge_input])?,
        tree.subtree("edge.entry.weights").read_slice::<f32>("weights", &[config.hidden, edge_input])?,
        tree.subtree("edge.entry").read_slice::<f32>("biases", &[config.hidden])?,
        tree.subtree("edge.block_norm").read_slice::<f32>("scales", &[config.hidden])?,
        tree.subtree("edge.block_norm").read_slice::<f32>("biases", &[config.hidden])?,
        tree.subtree("edge.up.weights").read_slice::<f32>("weights", &[config.edge_hidden, config.hidden])?,
        tree.subtree("edge.up").read_slice::<f32>("biases", &[config.edge_hidden])?,
        tree.subtree("edge.down.weights").read_slice::<f32>("weights", &[config.hidden, config.edge_hidden])?,
        tree.subtree("edge.down").read_slice::<f32>("biases", &[config.hidden])?,
        tree.subtree("edge.final_norm").read_slice::<f32>("scales", &[config.hidden])?,
        tree.subtree("edge.final_norm").read_slice::<f32>("biases", &[config.hidden])?,
        tree.subtree("edge.output.weights").read_slice::<f32>("weights", &[1, config.hidden])?,
        tree.subtree("edge.output").read_slice::<f32>("biases", &[1])?,
        tree.subtree("output.weights").read_slice::<f32>("weights", &[CONF_HEAD_NODE_OUTPUTS as u32, config.hidden])?,
        tree.subtree("output").read_slice::<f32>("biases", &[CONF_HEAD_NODE_OUTPUTS as u32])?,
    ]
    .concat();
    Ok(weights)
}
