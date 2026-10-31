use crate::{
    backends::common::{
        Backend,
        gpu_types::confidence_head::{POOL_INPUTS, SLOT_NUMBERS},
    },
    config::confidence_head::ConfidenceHeadConfig,
    encodable_block::confidence_head::{ConfidenceHeadError, DATA_TYPE},
    safetensors::ParameterTree,
};

pub struct NodeWeights<B: Backend> {
    pub pool_entry_weights: B::GlobalGpuBuffer,
    pub pool_entry_biases: B::GlobalGpuBuffer,
    pub pool_hidden_weights: B::GlobalGpuBuffer,
    pub pool_hidden_biases: B::GlobalGpuBuffer,
    pub pool_out_weights: B::GlobalGpuBuffer,
    pub pool_out_biases: B::GlobalGpuBuffer,
    pub slot_mean: B::GlobalGpuBuffer,
    pub slot_std: B::GlobalGpuBuffer,
}

impl<B: Backend> NodeWeights<B> {
    pub async fn new(
        config: &ConfidenceHeadConfig,
        parameter_tree: &ParameterTree<'_, B>,
    ) -> Result<Self, ConfidenceHeadError<B>> {
        let (width, pool) = (config.model_dim, config.pool_hidden_dim);
        let read = async |name: &str, shape: &[u32]| parameter_tree.read_buffer(name, shape, DATA_TYPE).await;
        let pool_entry_weights = read("pool_set.entry.weights", &[pool, POOL_INPUTS]).await?;
        let pool_entry_biases = read("pool_set.entry.biases", &[pool]).await?;
        let pool_hidden_weights = read("pool_set.hidden.weights", &[pool, pool]).await?;
        let pool_hidden_biases = read("pool_set.hidden.biases", &[pool]).await?;
        let pool_out_weights = read("pool_set.out.weights", &[width, 2 * pool]).await?;
        let pool_out_biases = read("pool_set.out.biases", &[width]).await?;
        let slot_mean = read("slot.mean", &[SLOT_NUMBERS]).await?;
        let slot_std = read("slot.std", &[SLOT_NUMBERS]).await?;
        Ok(Self {
            pool_entry_weights,
            pool_entry_biases,
            pool_hidden_weights,
            pool_hidden_biases,
            pool_out_weights,
            pool_out_biases,
            slot_mean,
            slot_std,
        })
    }
}
