use crate::{
    backends::common::{Backend, gpu_types::confidence_head::SLOT_NUMBERS},
    config::confidence_head::ConfidenceHeadConfig,
    encodable_block::confidence_head::{ConfidenceHeadError, DATA_TYPE},
    safetensors::ParameterTree,
};

pub struct EdgeWeights<B: Backend> {
    pub input_norm_scales: B::GlobalGpuBuffer,
    pub input_norm_biases: B::GlobalGpuBuffer,
    pub entry_weights: B::GlobalGpuBuffer,
    pub entry_biases: B::GlobalGpuBuffer,
    pub block_norm_scales: B::GlobalGpuBuffer,
    pub block_norm_biases: B::GlobalGpuBuffer,
    pub up_weights: B::GlobalGpuBuffer,
    pub up_biases: B::GlobalGpuBuffer,
    pub down_weights: B::GlobalGpuBuffer,
    pub down_biases: B::GlobalGpuBuffer,
    pub final_norm_scales: B::GlobalGpuBuffer,
    pub final_norm_biases: B::GlobalGpuBuffer,
    pub output_weights: B::GlobalGpuBuffer,
    pub output_biases: B::GlobalGpuBuffer,
    pub alpha_beta_weights: B::GlobalGpuBuffer,
    pub alpha_beta_biases: B::GlobalGpuBuffer,
}

impl<B: Backend> EdgeWeights<B> {
    pub async fn new(
        config: &ConfidenceHeadConfig,
        parameter_tree: &ParameterTree<'_, B>,
    ) -> Result<Self, ConfidenceHeadError<B>> {
        let (width, hidden) = (config.model_dim, config.edge_hidden_dim);
        let inputs = 3 * width + SLOT_NUMBERS;
        let read = async |name: &str, shape: &[u32]| parameter_tree.read_buffer(name, shape, DATA_TYPE).await;
        let input_norm_scales = read("edge.input_norm.scales", &[inputs]).await?;
        let input_norm_biases = read("edge.input_norm.biases", &[inputs]).await?;
        let entry_weights = read("edge.entry.weights", &[width, inputs]).await?;
        let entry_biases = read("edge.entry.biases", &[width]).await?;
        let block_norm_scales = read("edge.block_norm.scales", &[width]).await?;
        let block_norm_biases = read("edge.block_norm.biases", &[width]).await?;
        let up_weights = read("edge.up.weights", &[hidden, width]).await?;
        let up_biases = read("edge.up.biases", &[hidden]).await?;
        let down_weights = read("edge.down.weights", &[width, hidden]).await?;
        let down_biases = read("edge.down.biases", &[width]).await?;
        let final_norm_scales = read("edge.final_norm.scales", &[width]).await?;
        let final_norm_biases = read("edge.final_norm.biases", &[width]).await?;
        let output_weights = read("edge.output.weights", &[1, width]).await?;
        let output_biases = read("edge.output.biases", &[1]).await?;
        let alpha_beta_weights = read("alpha_beta.weights", &[2, width]).await?;
        let alpha_beta_biases = read("alpha_beta.biases", &[2]).await?;
        Ok(Self {
            input_norm_scales,
            input_norm_biases,
            entry_weights,
            entry_biases,
            block_norm_scales,
            block_norm_biases,
            up_weights,
            up_biases,
            down_weights,
            down_biases,
            final_norm_scales,
            final_norm_biases,
            output_weights,
            output_biases,
            alpha_beta_weights,
            alpha_beta_biases,
        })
    }
}
