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
    input_norm_scales: B::GlobalGpuBuffer,
    input_norm_biases: B::GlobalGpuBuffer,
    entry_weights: B::GlobalGpuBuffer,
    entry_biases: B::GlobalGpuBuffer,
    block_norm_scales: B::GlobalGpuBuffer,
    block_norm_biases: B::GlobalGpuBuffer,
    up_weights: B::GlobalGpuBuffer,
    up_biases: B::GlobalGpuBuffer,
    down_weights: B::GlobalGpuBuffer,
    down_biases: B::GlobalGpuBuffer,
    final_norm_scales: B::GlobalGpuBuffer,
    final_norm_biases: B::GlobalGpuBuffer,
    output_weights: B::GlobalGpuBuffer,
    output_biases: B::GlobalGpuBuffer,
    alpha_beta_weights: B::GlobalGpuBuffer,
    alpha_beta_biases: B::GlobalGpuBuffer,
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
        let input_norm_scales = parameter_tree.read_buffer("edge.input_norm.scales", &[inputs], DATA_TYPE).await?;
        let input_norm_biases = parameter_tree.read_buffer("edge.input_norm.biases", &[inputs], DATA_TYPE).await?;
        let entry_weights = parameter_tree.read_buffer("edge.entry.weights", &[width, inputs], DATA_TYPE).await?;
        let entry_biases = parameter_tree.read_buffer("edge.entry.biases", &[width], DATA_TYPE).await?;
        let block_norm_scales = parameter_tree.read_buffer("edge.block_norm.scales", &[width], DATA_TYPE).await?;
        let block_norm_biases = parameter_tree.read_buffer("edge.block_norm.biases", &[width], DATA_TYPE).await?;
        let up_weights = parameter_tree.read_buffer("edge.up.weights", &[hidden, width], DATA_TYPE).await?;
        let up_biases = parameter_tree.read_buffer("edge.up.biases", &[hidden], DATA_TYPE).await?;
        let down_weights = parameter_tree.read_buffer("edge.down.weights", &[width, hidden], DATA_TYPE).await?;
        let down_biases = parameter_tree.read_buffer("edge.down.biases", &[width], DATA_TYPE).await?;
        let final_norm_scales = parameter_tree.read_buffer("edge.final_norm.scales", &[width], DATA_TYPE).await?;
        let final_norm_biases = parameter_tree.read_buffer("edge.final_norm.biases", &[width], DATA_TYPE).await?;
        let output_weights = parameter_tree.read_buffer("edge.output.weights", &[1, width], DATA_TYPE).await?;
        let output_biases = parameter_tree.read_buffer("edge.output.biases", &[1], DATA_TYPE).await?;
        let alpha_beta_weights = parameter_tree.read_buffer("alpha_beta.weights", &[2, width], DATA_TYPE).await?;
        let alpha_beta_biases = parameter_tree.read_buffer("alpha_beta.biases", &[2], DATA_TYPE).await?;
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
            steer: kernel(false)?,
            refine: kernel(true)?,
        })
    }

    pub fn encode(
        &self,
        node_metadata: Option<impl BufferRef<Backend = B>>,
        slot_rows: impl BufferRef<Backend = B>,
        nodes: impl BufferRef<Backend = B>,
        node_shifts: Option<impl BufferRef<Backend = B>>,
        child_shifts: Option<impl BufferRef<Backend = B>>,
        token_features: impl BufferRef<Backend = B>,
        child_numbers: impl BufferRef<Backend = B>,
        child_pools: impl BufferRef<Backend = B>,
        scores: impl BufferRef<Backend = B>,
        frontier: impl BufferMut<Backend = B>,
        packed_tree: impl BufferMut<Backend = B>,
        child_inputs: Option<impl BufferMut<Backend = B>>,
        params: ConfidenceHeadParams,
        groups: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let kernel = if node_shifts.is_some() {
            &self.refine
        } else {
            &self.steer
        };
        kernel.encode(
            node_metadata,
            slot_rows,
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
            &self.input_norm_scales,
            &self.input_norm_biases,
            &self.entry_weights,
            &self.entry_biases,
            &self.block_norm_scales,
            &self.block_norm_biases,
            &self.up_weights,
            &self.up_biases,
            &self.down_weights,
            &self.down_biases,
            &self.final_norm_scales,
            &self.final_norm_biases,
            &self.output_weights,
            &self.output_biases,
            &self.alpha_beta_weights,
            &self.alpha_beta_biases,
            params,
            groups,
            command_buffer,
        );
    }
}
