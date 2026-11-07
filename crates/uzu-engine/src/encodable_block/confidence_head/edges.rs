use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, Kernels,
        gpu_types::confidence_head::{ConfidenceHeadParams, SLOT_NUMBERS},
        kernel::ConfidenceHeadEdgesKernel,
    },
    config::confidence_head::ConfidenceHeadConfig,
    encodable_block::confidence_head::{ConfidenceHeadError, DATA_TYPE},
    safetensors::ParameterTree,
};

pub struct Edges<B: Backend> {
    steer: <B::Kernels as Kernels>::ConfidenceHeadEdgesKernel,
    refine: <B::Kernels as Kernels>::ConfidenceHeadEdgesKernel,
    slot_stats: B::GlobalGpuBuffer,
    input_norm: B::GlobalGpuBuffer,
    entry_weights: B::GlobalGpuBuffer,
    entry_biases: B::GlobalGpuBuffer,
    block_norm: B::GlobalGpuBuffer,
    up_weights: B::GlobalGpuBuffer,
    up_biases: B::GlobalGpuBuffer,
    down_weights: B::GlobalGpuBuffer,
    down_biases: B::GlobalGpuBuffer,
    final_norm: B::GlobalGpuBuffer,
    readout_weights: B::GlobalGpuBuffer,
    readout_biases: B::GlobalGpuBuffer,
}

impl<B: Backend> Edges<B> {
    pub async fn new(
        config: &ConfidenceHeadConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, ConfidenceHeadError<B>> {
        let (width, hidden) = (config.model_dim, config.edge_hidden_dim);
        let inputs = 3 * width + SLOT_NUMBERS;
        let kernel = |refine| {
            <B::Kernels as Kernels>::ConfidenceHeadEdgesKernel::new(context, width, hidden, refine)
                .map_err(ConfidenceHeadError::Backend)
        };
        let slot_stats = parameter_tree.read_buffer("slot.stats", &[2, SLOT_NUMBERS], DATA_TYPE).await?;
        let input_norm = parameter_tree.read_buffer("edge.input_norm", &[2, inputs], DATA_TYPE).await?;
        let entry_weights = parameter_tree.read_buffer("edge.entry.weights", &[width, inputs], DATA_TYPE).await?;
        let entry_biases = parameter_tree.read_buffer("edge.entry.biases", &[width], DATA_TYPE).await?;
        let block_norm = parameter_tree.read_buffer("edge.block_norm", &[2, width], DATA_TYPE).await?;
        let up_weights = parameter_tree.read_buffer("edge.up.weights", &[hidden, width], DATA_TYPE).await?;
        let up_biases = parameter_tree.read_buffer("edge.up.biases", &[hidden], DATA_TYPE).await?;
        let down_weights = parameter_tree.read_buffer("edge.down.weights", &[width, hidden], DATA_TYPE).await?;
        let down_biases = parameter_tree.read_buffer("edge.down.biases", &[width], DATA_TYPE).await?;
        let final_norm = parameter_tree.read_buffer("edge.final_norm", &[2, width], DATA_TYPE).await?;
        let readout_weights = parameter_tree.read_buffer("edge.readout.weights", &[3, width], DATA_TYPE).await?;
        let readout_biases = parameter_tree.read_buffer("edge.readout.biases", &[3], DATA_TYPE).await?;
        Ok(Self {
            slot_stats,
            input_norm,
            entry_weights,
            entry_biases,
            block_norm,
            up_weights,
            up_biases,
            down_weights,
            down_biases,
            final_norm,
            readout_weights,
            readout_biases,
            steer: kernel(false)?,
            refine: kernel(true)?,
        })
    }

    pub fn encode(
        &self,
        node_metadata: Option<impl BufferRef<Backend = B>>,
        node_valid: Option<impl BufferRef<Backend = B>>,
        candidate_ids: Option<impl BufferRef<Backend = B>>,
        pool_features: Option<impl BufferRef<Backend = B>>,
        slot_rows: impl BufferMut<Backend = B>,
        row_slots: Option<impl BufferMut<Backend = B>>,
        nodes: impl BufferRef<Backend = B>,
        node_shifts: Option<impl BufferRef<Backend = B>>,
        child_shifts: Option<impl BufferRef<Backend = B>>,
        token_features: impl BufferRef<Backend = B>,
        child_numbers: impl BufferMut<Backend = B>,
        child_pools: impl BufferMut<Backend = B>,
        scores: impl BufferRef<Backend = B>,
        frontier: impl BufferMut<Backend = B>,
        packed_tree: impl BufferMut<Backend = B>,
        child_inputs: Option<impl BufferMut<Backend = B>>,
        params: ConfidenceHeadParams,
        groups: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let refine = node_shifts.is_some();
        let kernel = if refine {
            &self.refine
        } else {
            &self.steer
        };
        kernel.encode(
            node_metadata,
            node_valid,
            candidate_ids,
            pool_features,
            (!refine).then_some(&self.slot_stats),
            slot_rows,
            row_slots,
            nodes,
            node_shifts,
            child_shifts,
            token_features,
            child_numbers,
            child_pools,
            scores,
            frontier,
            packed_tree,
            child_inputs,
            &self.input_norm,
            &self.entry_weights,
            &self.entry_biases,
            &self.block_norm,
            &self.up_weights,
            &self.up_biases,
            &self.down_weights,
            &self.down_biases,
            &self.final_norm,
            &self.readout_weights,
            &self.readout_biases,
            params,
            groups,
            command_buffer,
        );
    }
}
