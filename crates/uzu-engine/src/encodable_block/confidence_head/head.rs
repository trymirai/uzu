use crate::{
    backends::common::{
        Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, CommandBufferExecutable,
        CommandBufferPending, Context, Kernels,
        gpu_types::{
            confidence_head::{
                CHILDREN_MAX, ConfidenceHeadParams, EDGE_HIDDEN_MAX, MODEL_MAX, POOL_FEATURES, POOL_HIDDEN_MAX,
                SLOT_NUMBERS,
            },
            weaver::{CANDIDATES_MAX, MetadataIdx},
        },
        kernel::ConfidenceHeadPoolsKernel,
    },
    config::confidence_head::ConfidenceHeadConfig,
    data_type::DataType,
    encodable_block::{
        confidence_head::{ConfidenceHeadError, DATA_TYPE, Edges, Nodes, TreeAttention},
        embedding::Embedding,
        linear::Linear,
        normalization::{Normalization, PostLayerScalar, ShortcutMode},
    },
    safetensors::ParameterTree,
    utils::size_for_shape::size_for_shape,
};

const TOKEN_CHUNK: u32 = 2048;

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
    round_metadata: B::ScratchBuffer,
    round_valid: B::ScratchBuffer,
    params: ConfidenceHeadParams,
}

pub struct ConfidenceHead<B: Backend> {
    config: ConfidenceHeadConfig,
    draft_norm: Normalization<B>,
    draft_projection: Box<dyn Linear<B>>,
    token_features: B::GlobalGpuBuffer,
    pools: <B::Kernels as Kernels>::ConfidenceHeadPoolsKernel,
    nodes: Nodes<B>,
    edges: Edges<B>,
    tree_attention: Option<TreeAttention<B>>,
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
        if width > MODEL_MAX
            || config.pool_hidden_dim > POOL_HIDDEN_MAX
            || config.edge_hidden_dim > EDGE_HIDDEN_MAX
            || !width.is_multiple_of(4)
            || !config.pool_hidden_dim.is_multiple_of(2)
            || !config.edge_hidden_dim.is_multiple_of(4)
            || config.max_depth < tree_depth
        {
            return Err(ConfidenceHeadError::UnsupportedShape(format!(
                "width {width}, pool width {}, edge width {} and depth {} for trees of depth {tree_depth}",
                config.pool_hidden_dim, config.edge_hidden_dim, config.max_depth
            )));
        }
        let target_dim = target_embedding.model_dim();
        let norm_config = &config.norm_config;
        let draft_norm = Normalization::new(
            target_dim,
            DATA_TYPE,
            DATA_TYPE,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            norm_config,
            &parameter_tree.subtree("draft_norm"),
            context,
        )
        .await?;
        let draft_projection = <dyn Linear<B>>::new(
            target_dim,
            width,
            DATA_TYPE,
            true,
            &parameter_tree.subtree("draft_projection"),
            context,
        )
        .await?;
        let token_norm = Normalization::new(
            target_dim,
            DATA_TYPE,
            DATA_TYPE,
            None,
            ShortcutMode::None,
            PostLayerScalar::None,
            norm_config,
            &parameter_tree.subtree("token_norm"),
            context,
        )
        .await?;
        let token_projection = <dyn Linear<B>>::new(
            target_dim,
            width,
            DATA_TYPE,
            true,
            &parameter_tree.subtree("token_projection"),
            context,
        )
        .await?;
        let token_features =
            encode_token_features(&token_norm, &*token_projection, target_embedding, width, context).await?;
        let nodes = Nodes::new(config, parameter_tree, context).await?;
        let edges = Edges::new(config, parameter_tree, context).await?;
        let tree_attention = match &config.tree_attention_config {
            Some(tree_config) => {
                Some(TreeAttention::new(config, tree_config, &parameter_tree.subtree("tree"), context).await?)
            },
            None => None,
        };
        Ok(Self {
            config: config.clone(),
            draft_norm,
            draft_projection,
            token_features,
            nodes,
            edges,
            tree_attention,
            pools: <B::Kernels as Kernels>::ConfidenceHeadPoolsKernel::new(context)
                .map_err(ConfidenceHeadError::Backend)?,
        })
    }

    pub fn encode_step(
        &self,
        draft_hidden: impl BufferRef<Backend = B>,
        candidate_logits: impl BufferRef<Backend = B>,
        candidate_ids: impl BufferRef<Backend = B>,
        depth_seeds: impl BufferRef<Backend = B>,
        depths: u32,
        candidates: u32,
        children: u32,
        tree_slot_count: u32,
        vocab_size: u32,
        round_rows: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<ConfidenceHeadStep<B>, ConfidenceHeadError<B>> {
        let params = ConfidenceHeadParams {
            candidates,
            children,
            tree_slot_count,
            vocab_size,
            logit_cap: self.config.logit_cap,
            epsilon: self.config.norm_config.epsilon,
        };
        if candidates > CANDIDATES_MAX || children > CHILDREN_MAX {
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
        let nodes = scratch(&[slots, width], DATA_TYPE)?;
        let node_inputs = scratch(&[slots, width], DATA_TYPE)?;
        let child_inputs = scratch(&[entries, width + SLOT_NUMBERS], DATA_TYPE)?;
        let child_pools = scratch(&[entries], DataType::U32)?;
        let child_numbers = scratch(&[entries, SLOT_NUMBERS], DataType::F32)?;
        let mut slot_rows = scratch(&[slots], DataType::U32)?;
        let mut row_slots = scratch(&[1 + entries], DataType::U32)?;
        let round_metadata = scratch(&[MetadataIdx::COUNT as u32, round_rows], DataType::U32)?;
        let round_valid = scratch(&[round_rows], DataType::U32)?;
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
            round_metadata,
            round_valid,
            params,
        })
    }

    pub fn encode_rows(
        &self,
        step: &mut ConfidenceHeadStep<B>,
        node_metadata: impl BufferRef<Backend = B>,
        node_valid: impl BufferRef<Backend = B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        command_buffer.encode_copy(node_metadata, &mut step.round_metadata);
        command_buffer.encode_copy(node_valid, &mut step.round_valid);
    }

    pub fn encode_round(
        &self,
        step: &mut ConfidenceHeadStep<B>,
        residual_logits: impl BufferRef<Backend = B>,
        candidate_logits: impl BufferRef<Backend = B>,
        candidate_ids: impl BufferRef<Backend = B>,
        packed_tree: impl BufferMut<Backend = B>,
        frontier: impl BufferMut<Backend = B>,
        rows: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) {
        self.nodes.encode(
            residual_logits,
            candidate_logits,
            &step.pool_features,
            &step.round_metadata,
            &step.round_valid,
            &step.draft,
            &mut step.scores,
            &mut step.nodes,
            &mut step.node_inputs,
            step.params,
            rows,
            command_buffer,
        );
        self.edges.encode(
            Some(&step.round_metadata),
            Some(&step.round_valid),
            Some(candidate_ids),
            Some(&step.pool_features),
            &mut step.slot_rows,
            Some(&mut step.row_slots),
            &step.nodes,
            None::<&B::ScratchBuffer>,
            None::<&B::ScratchBuffer>,
            &self.token_features,
            &mut step.child_numbers,
            &mut step.child_pools,
            &step.scores,
            frontier,
            packed_tree,
            Some(&mut step.child_inputs),
            step.params,
            rows,
            command_buffer,
        );
    }

    pub fn encode_prune(
        &self,
        mut step: ConfidenceHeadStep<B>,
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
        self.edges.encode(
            None::<&B::ScratchBuffer>,
            None::<&B::ScratchBuffer>,
            None::<&B::ScratchBuffer>,
            None::<&B::ScratchBuffer>,
            &mut step.slot_rows,
            None::<&mut B::ScratchBuffer>,
            &step.nodes,
            Some(&node_shifts),
            Some(&child_shifts),
            &self.token_features,
            &mut step.child_numbers,
            &mut step.child_pools,
            &step.scores,
            frontier.reborrow(),
            packed_tree.reborrow(),
            None::<&mut B::ScratchBuffer>,
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
    let row_bytes = size_for_shape(&[width], DATA_TYPE);
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
