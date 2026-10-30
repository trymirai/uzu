use crate::{
    backends::common::Backend,
    config::confidence_head::ConfidenceHeadConfig,
    data_type::DataType,
    encodable_block::confidence_head::{ConfidenceHeadError, SLOT_NUMBERS},
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
        let read = async |name: &str, shape: &[u32]| parameter_tree.read_buffer(name, shape, DataType::BF16).await;
        let pool_entry_weights = read("pool_set.entry.weights", &[pool, 3]).await?;
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
