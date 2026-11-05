use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, Kernels,
        gpu_types::confidence_head::{ConfidenceHeadParams, POOL_INPUTS, SLOT_NUMBERS},
        kernel::ConfidenceHeadNodesKernel,
    },
    config::confidence_head::ConfidenceHeadConfig,
    encodable_block::confidence_head::{ConfidenceHeadError, DATA_TYPE},
    safetensors::ParameterTree,
};

pub struct Nodes<B: Backend> {
    kernel: <B::Kernels as Kernels>::ConfidenceHeadNodesKernel,
    pool_entry_weights: B::GlobalGpuBuffer,
    pool_entry_biases: B::GlobalGpuBuffer,
    pool_hidden_weights: B::GlobalGpuBuffer,
    pool_hidden_biases: B::GlobalGpuBuffer,
    pool_out_weights: B::GlobalGpuBuffer,
    pool_out_biases: B::GlobalGpuBuffer,
    slot_mean: B::GlobalGpuBuffer,
    slot_std: B::GlobalGpuBuffer,
}

impl<B: Backend> Nodes<B> {
    pub async fn new(
        config: &ConfidenceHeadConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, ConfidenceHeadError<B>> {
        let (width, pool) = (config.model_dim, config.pool_hidden_dim);
        let pool_entry_weights =
            parameter_tree.read_buffer("pool_set.entry.weights", &[pool, POOL_INPUTS], DATA_TYPE).await?;
        let pool_entry_biases = parameter_tree.read_buffer("pool_set.entry.biases", &[pool], DATA_TYPE).await?;
        let pool_hidden_weights =
            parameter_tree.read_buffer("pool_set.hidden.weights", &[pool, pool], DATA_TYPE).await?;
        let pool_hidden_biases = parameter_tree.read_buffer("pool_set.hidden.biases", &[pool], DATA_TYPE).await?;
        let pool_out_weights =
            parameter_tree.read_buffer("pool_set.out.weights", &[width, 2 * pool], DATA_TYPE).await?;
        let pool_out_biases = parameter_tree.read_buffer("pool_set.out.biases", &[width], DATA_TYPE).await?;
        let slot_mean = parameter_tree.read_buffer("slot.mean", &[SLOT_NUMBERS], DATA_TYPE).await?;
        let slot_std = parameter_tree.read_buffer("slot.std", &[SLOT_NUMBERS], DATA_TYPE).await?;
        Ok(Self {
            pool_entry_weights,
            pool_entry_biases,
            pool_hidden_weights,
            pool_hidden_biases,
            pool_out_weights,
            pool_out_biases,
            slot_mean,
            slot_std,
            kernel: <B::Kernels as Kernels>::ConfidenceHeadNodesKernel::new(context, width, pool)
                .map_err(ConfidenceHeadError::Backend)?,
        })
    }

    pub fn encode(
        &self,
        residual_logits: impl BufferRef<Backend = B>,
        candidate_logits: impl BufferRef<Backend = B>,
        candidate_ids: impl BufferRef<Backend = B>,
        pool_features: impl BufferRef<Backend = B>,
        node_metadata: impl BufferRef<Backend = B>,
        node_valid: impl BufferRef<Backend = B>,
        packed_tree: impl BufferRef<Backend = B>,
        frontier: impl BufferRef<Backend = B>,
        draft: impl BufferRef<Backend = B>,
        scores: impl BufferMut<Backend = B>,
        nodes: impl BufferMut<Backend = B>,
        node_inputs: impl BufferMut<Backend = B>,
        child_pools: impl BufferMut<Backend = B>,
        child_numbers: impl BufferMut<Backend = B>,
        slot_rows: impl BufferMut<Backend = B>,
        row_slots: impl BufferMut<Backend = B>,
        params: ConfidenceHeadParams,
        rows: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        self.kernel.encode(
            residual_logits,
            candidate_logits,
            candidate_ids,
            pool_features,
            node_metadata,
            node_valid,
            packed_tree,
            frontier,
            draft,
            &self.pool_entry_weights,
            &self.pool_entry_biases,
            &self.pool_hidden_weights,
            &self.pool_hidden_biases,
            &self.pool_out_weights,
            &self.pool_out_biases,
            &self.slot_mean,
            &self.slot_std,
            scores,
            nodes,
            node_inputs,
            child_pools,
            child_numbers,
            slot_rows,
            row_slots,
            params,
            rows,
            command_buffer,
        );
    }
}
