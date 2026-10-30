use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferPending, Context, Kernels,
        gpu_types::confidence_head::ConfidenceHeadParams,
        kernel::{ConfidenceHeadEdgesKernel, ConfidenceHeadNodesKernel, ConfidenceHeadPoolsKernel},
    },
    config::confidence_head::ConfidenceHeadConfig,
    data_type::DataType,
    encodable_block::{
        confidence_head::{
            ConfidenceHeadError, SLOT_NUMBERS, TreeAttention, edge_weights::EdgeWeights, linear,
            node_weights::NodeWeights, norm,
        },
        embedding::Embedding,
        linear::Linear,
        normalization::{Normalization, ShortcutMode},
    },
    safetensors::ParameterTree,
    utils::size_for_shape::size_for_shape,
};

const TOKEN_CHUNK: u32 = 2048;
const POOL_FEATURES: u32 = 3;

pub struct ConfidenceHeadStep<B: Backend> {
    draft: B::ScratchBuffer,
    pool_features: B::ScratchBuffer,
    scores: B::ScratchBuffer,
    nodes: B::ScratchBuffer,
    node_inputs: B::ScratchBuffer,
    child_inputs: B::ScratchBuffer,
    child_pools: B::ScratchBuffer,
    child_numbers: B::ScratchBuffer,
    slot_rows: B::ScratchBuffer,
    row_slots: B::ScratchBuffer,
    params: ConfidenceHeadParams,
}

pub struct ConfidenceHead<B: Backend> {
    config: ConfidenceHeadConfig,
    draft_norm: Normalization<B>,
    draft_projection: Box<dyn Linear<B>>,
    node_weights: NodeWeights<B>,
    edge_weights: EdgeWeights<B>,
    token_features: B::GlobalGpuBuffer,
    tree_attention: Option<TreeAttention<B>>,
    pools: <B::Kernels as Kernels>::ConfidenceHeadPoolsKernel,
    nodes: <B::Kernels as Kernels>::ConfidenceHeadNodesKernel,
    edges: <B::Kernels as Kernels>::ConfidenceHeadEdgesKernel,
    refine: <B::Kernels as Kernels>::ConfidenceHeadEdgesKernel,
}

impl<B: Backend> ConfidenceHead<B> {
    pub async fn new(
        config: &ConfidenceHeadConfig,
        parameter_tree: &ParameterTree<'_, B>,
        target_embedding: &Embedding<B>,
        tree_depth: u32,
        context: &B::Context,
    ) -> Result<Self, ConfidenceHeadError<B>> {
        let width = config.model_dim;
        if width > 64 || config.pool_hidden_dim > 16 || config.edge_hidden_dim > 128 || config.max_depth < tree_depth {
            return Err(ConfidenceHeadError::UnsupportedShape(format!(
                "width {width}, pool width {}, edge width {} and depth {} for trees of depth {tree_depth}",
                config.pool_hidden_dim, config.edge_hidden_dim, config.max_depth
            )));
        }
        let target_dim = target_embedding.model_dim();
        let norm_config = &config.norm_config;
        let draft_norm =
            norm(parameter_tree, "draft_norm", target_dim, ShortcutMode::None, norm_config, context).await?;
        let draft_projection = linear(parameter_tree, "draft_projection", target_dim, width, context).await?;
        let token_norm =
            norm(parameter_tree, "token_norm", target_dim, ShortcutMode::None, norm_config, context).await?;
        let token_projection = linear(parameter_tree, "token_projection", target_dim, width, context).await?;
        let token_features =
            encode_token_features(&token_norm, &*token_projection, target_embedding, width, context).await?;
        let node_weights = NodeWeights::new(config, parameter_tree).await?;
        let edge_weights = EdgeWeights::new(config, parameter_tree).await?;
        let tree_attention = match &config.tree_attention_config {
            Some(tree_config) => {
                Some(TreeAttention::new(config, tree_config, &parameter_tree.subtree("tree"), context).await?)
            },
            None => None,
        };
        let edges = |refine| {
            <B::Kernels as Kernels>::ConfidenceHeadEdgesKernel::new(context, width, config.edge_hidden_dim, refine)
                .map_err(ConfidenceHeadError::Backend)
        };
        Ok(Self {
            config: config.clone(),
            draft_norm,
            draft_projection,
            node_weights,
            edge_weights,
            token_features,
            tree_attention,
            pools: <B::Kernels as Kernels>::ConfidenceHeadPoolsKernel::new(context)
                .map_err(ConfidenceHeadError::Backend)?,
            nodes: <B::Kernels as Kernels>::ConfidenceHeadNodesKernel::new(context, width, config.pool_hidden_dim)
                .map_err(ConfidenceHeadError::Backend)?,
            edges: edges(false)?,
            refine: edges(true)?,
        })
    }

    pub fn params(
        &self,
        candidates: u32,
        children: u32,
        tree_slot_count: u32,
        vocab_size: u32,
    ) -> ConfidenceHeadParams {
        ConfidenceHeadParams {
            candidates,
            children,
            tree_slot_count,
            vocab_size,
            logit_cap: self.config.logit_cap,
            epsilon: self.config.norm_config.epsilon,
        }
    }

    pub fn encode_step(
        &self,
        draft_hidden: impl BufferRef<Backend = B>,
        candidate_logits: impl BufferRef<Backend = B>,
        candidate_ids: impl BufferRef<Backend = B>,
        depth_seeds: impl BufferRef<Backend = B>,
        depths: u32,
        params: ConfidenceHeadParams,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<ConfidenceHeadStep<B>, ConfidenceHeadError<B>> {
        if params.candidates > 512 || params.children > 4 {
            return Err(ConfidenceHeadError::UnsupportedShape(format!(
                "{} candidates and {} children",
                params.candidates, params.children
            )));
        }
        let normalized = self
            .draft_norm
            .encode(draft_hidden, 1, depths, None::<&mut B::ScratchBuffer>, command_buffer)
            .map_err(ConfidenceHeadError::Backend)?;
        let draft = self.draft_projection.encode(normalized, depths, command_buffer)?;
        let (slots, width) = (params.tree_slot_count, self.config.model_dim);
        let entries = slots * params.children;
        let mut scratch = |shape: &[u32], data_type: DataType| {
            command_buffer.allocate_scratch_for_shape(shape, data_type).map_err(ConfidenceHeadError::Backend)
        };
        let mut pool_features = scratch(&[depths, POOL_FEATURES, params.candidates], DataType::F32)?;
        let scores = scratch(&[slots, 2, params.candidates], DataType::F32)?;
        let nodes = scratch(&[slots, width], DataType::BF16)?;
        let node_inputs = scratch(&[slots, width], DataType::BF16)?;
        let child_inputs = scratch(&[entries, width + SLOT_NUMBERS], DataType::BF16)?;
        let child_pools = scratch(&[entries], DataType::U32)?;
        let child_numbers = scratch(&[entries, SLOT_NUMBERS], DataType::F32)?;
        let mut slot_rows = scratch(&[slots], DataType::U32)?;
        let mut row_slots = scratch(&[1 + entries], DataType::U32)?;
        command_buffer.encode_fill(&mut slot_rows, 0xff);
        command_buffer.encode_fill(&mut row_slots, 0xff);
        self.pools.encode(
            candidate_logits,
            candidate_ids,
            depth_seeds,
            &mut pool_features,
            params,
            depths,
            command_buffer,
        );
        Ok(ConfidenceHeadStep {
            draft,
            pool_features,
            scores,
            nodes,
            node_inputs,
            child_inputs,
            child_pools,
            child_numbers,
            slot_rows,
            row_slots,
            params,
        })
    }

    pub fn encode_round(
        &self,
        step: &mut ConfidenceHeadStep<B>,
        residual_logits: impl BufferRef<Backend = B>,
        candidate_logits: impl BufferRef<Backend = B>,
        candidate_ids: impl BufferRef<Backend = B>,
        node_metadata: impl BufferRef<Backend = B>,
        node_valid: impl BufferRef<Backend = B>,
        mut packed_tree: impl BufferMut<Backend = B>,
        mut frontier: impl BufferMut<Backend = B>,
        rows: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        let weights = &self.node_weights;
        self.nodes.encode(
            residual_logits,
            candidate_logits,
            candidate_ids,
            &step.pool_features,
            node_metadata,
            node_valid,
            packed_tree.as_ref(),
            frontier.as_ref(),
            &step.draft,
            &weights.pool_entry_weights,
            &weights.pool_entry_biases,
            &weights.pool_hidden_weights,
            &weights.pool_hidden_biases,
            &weights.pool_out_weights,
            &weights.pool_out_biases,
            &weights.slot_mean,
            &weights.slot_std,
            &mut step.scores,
            &mut step.nodes,
            &mut step.node_inputs,
            &mut step.child_pools,
            &mut step.child_numbers,
            &mut step.slot_rows,
            &mut step.row_slots,
            step.params,
            rows,
            command_buffer,
        );
        let weights = &self.edge_weights;
        self.edges.encode(
            Some(node_metadata),
            &step.slot_rows,
            &step.nodes,
            None::<&B::ScratchBuffer>,
            None::<&B::ScratchBuffer>,
            &self.token_features,
            &step.child_numbers,
            &step.child_pools,
            &step.scores,
            frontier.reborrow(),
            packed_tree.reborrow(),
            Some(&mut step.child_inputs),
            &weights.input_norm_scales,
            &weights.input_norm_biases,
            &weights.entry_weights,
            &weights.entry_biases,
            &weights.block_norm_scales,
            &weights.block_norm_biases,
            &weights.up_weights,
            &weights.up_biases,
            &weights.down_weights,
            &weights.down_biases,
            &weights.final_norm_scales,
            &weights.final_norm_biases,
            &weights.output_weights,
            &weights.output_biases,
            &weights.alpha_beta_weights,
            &weights.alpha_beta_biases,
            step.params,
            rows,
            command_buffer,
        );
    }

    pub fn encode_prune(
        &self,
        step: ConfidenceHeadStep<B>,
        mut packed_tree: impl BufferMut<Backend = B>,
        mut frontier: impl BufferMut<Backend = B>,
        slot_ancestors: impl BufferRef<Backend = B>,
        ancestor_stride: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), ConfidenceHeadError<B>> {
        let Some(tree_attention) = &self.tree_attention else {
            return Ok(());
        };
        let slots = step.params.tree_slot_count;
        let (node_shifts, child_shifts) = tree_attention.encode(
            step.node_inputs,
            step.child_inputs,
            frontier.as_ref(),
            &step.row_slots,
            &step.slot_rows,
            slot_ancestors,
            ancestor_stride,
            slots,
            1 + slots * step.params.children,
            command_buffer,
        )?;
        let weights = &self.edge_weights;
        self.refine.encode(
            None::<&B::ScratchBuffer>,
            &step.slot_rows,
            &step.nodes,
            Some(&node_shifts),
            Some(&child_shifts),
            &self.token_features,
            &step.child_numbers,
            &step.child_pools,
            &step.scores,
            frontier.reborrow(),
            packed_tree.reborrow(),
            None::<&mut B::ScratchBuffer>,
            &weights.input_norm_scales,
            &weights.input_norm_biases,
            &weights.entry_weights,
            &weights.entry_biases,
            &weights.block_norm_scales,
            &weights.block_norm_biases,
            &weights.up_weights,
            &weights.up_biases,
            &weights.down_weights,
            &weights.down_biases,
            &weights.final_norm_scales,
            &weights.final_norm_biases,
            &weights.output_weights,
            &weights.output_biases,
            &weights.alpha_beta_weights,
            &weights.alpha_beta_biases,
            step.params,
            slots,
            command_buffer,
        );
        Ok(())
    }
}

async fn encode_token_features<B: Backend>(
    token_norm: &Normalization<B>,
    token_projection: &dyn Linear<B>,
    target_embedding: &Embedding<B>,
    width: u32,
    context: &B::Context,
) -> Result<B::GlobalGpuBuffer, ConfidenceHeadError<B>> {
    let vocab_size = target_embedding.vocab_size();
    let row_bytes = size_for_shape(&[width], DataType::BF16);
    let mut table = context.create_buffer_gpu(vocab_size as usize * row_bytes).map_err(ConfidenceHeadError::Backend)?;
    for start in (0..vocab_size).step_by(TOKEN_CHUNK as usize) {
        let count = TOKEN_CHUNK.min(vocab_size - start);
        let mut command_buffer = context
            .create_command_buffer(Some("confidence head tokens"), None)
            .map_err(ConfidenceHeadError::Backend)?;
        let ids = (start..start + count).collect::<Box<[u32]>>();
        let ids = command_buffer.allocate_constant_from_slice(&ids).map_err(ConfidenceHeadError::Backend)?;
        let rows = target_embedding.encode_lookup(&ids, count, &mut command_buffer)?;
        drop(ids);
        let normalized = token_norm
            .encode(&rows, 0, count, None::<&mut B::ScratchBuffer>, &mut command_buffer)
            .map_err(ConfidenceHeadError::Backend)?;
        let projected = token_projection.encode(normalized, count, &mut command_buffer)?;
        let range = start as usize * row_bytes..(start + count) as usize * row_bytes;
        command_buffer.encode_copy(&projected, (&mut table).subrange_mut(range));
        command_buffer.end_encoding().submit().wait_until_completed().await.map_err(ConfidenceHeadError::Backend)?;
    }
    Ok(table)
}

#[cfg(test)]
#[path = "../../../unit/encodable_block/confidence_head/head_test.rs"]
mod tests;
