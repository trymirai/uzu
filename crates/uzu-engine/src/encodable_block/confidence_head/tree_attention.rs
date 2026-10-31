use crate::{
    backends::common::{
        Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
        gpu_types::{
            activation_type::ActivationType,
            confidence_head::{RELATIONS, SLOT_NUMBERS, TREE_ROWS_MAX},
        },
        kernel::{
            ActivationKernel, ConfidenceHeadTreeAttentionKernel, ConfidenceHeadTreeInputKernel,
            ConfidenceHeadTreeRelationsKernel,
        },
    },
    config::{
        confidence_head::ConfidenceHeadConfig, confidence_head_tree_attention::ConfidenceHeadTreeAttentionConfig,
    },
    data_type::DataType,
    encodable_block::{
        confidence_head::{ConfidenceHeadError, DATA_TYPE, linear, norm},
        linear::Linear,
        normalization::{Normalization, ShortcutMode},
    },
    safetensors::ParameterTree,
};

struct TreeAttentionLayer<B: Backend> {
    attention_norm: Normalization<B>,
    qkv: Box<dyn Linear<B>>,
    out: Box<dyn Linear<B>>,
    mlp_norm: Normalization<B>,
    up: Box<dyn Linear<B>>,
    down: Box<dyn Linear<B>>,
}

pub struct TreeAttention<B: Backend> {
    config: ConfidenceHeadTreeAttentionConfig,
    node_in: Box<dyn Linear<B>>,
    child_in: Box<dyn Linear<B>>,
    depth_embedding: B::GlobalGpuBuffer,
    relation_bias: B::GlobalGpuBuffer,
    layers: Box<[TreeAttentionLayer<B>]>,
    final_norm: Normalization<B>,
    couple_node: Box<dyn Linear<B>>,
    couple_child: Box<dyn Linear<B>>,
    activation: <B::Kernels as Kernels>::ActivationKernel,
    input: <B::Kernels as Kernels>::ConfidenceHeadTreeInputKernel,
    relations: <B::Kernels as Kernels>::ConfidenceHeadTreeRelationsKernel,
    attention: <B::Kernels as Kernels>::ConfidenceHeadTreeAttentionKernel,
}

impl<B: Backend> TreeAttention<B> {
    pub async fn new(
        head_config: &ConfidenceHeadConfig,
        config: &ConfidenceHeadTreeAttentionConfig,
        parameter_tree: &ParameterTree<'_, B>,
        context: &B::Context,
    ) -> Result<Self, ConfidenceHeadError<B>> {
        let width = config.model_dim;
        let norm_config = &head_config.norm_config;
        let mut layers = Vec::with_capacity(config.num_layers as usize);
        for index in 0..config.num_layers {
            let layer = parameter_tree.subtree("layers").subtree(&index.to_string());
            let first = if index == 0 {
                ShortcutMode::Copy
            } else {
                ShortcutMode::Add
            };
            let attention_norm = norm(&layer, "attention_norm", width, first, norm_config, context).await?;
            let qkv = linear(&layer, "qkv", width, 3 * width, context).await?;
            let out = linear(&layer, "out", width, width, context).await?;
            let mlp_norm = norm(&layer, "mlp_norm", width, ShortcutMode::Add, norm_config, context).await?;
            let up = linear(&layer, "up", width, config.hidden_dim, context).await?;
            let down = linear(&layer, "down", config.hidden_dim, width, context).await?;
            layers.push(TreeAttentionLayer {
                attention_norm,
                qkv,
                out,
                mlp_norm,
                up,
                down,
            });
        }
        let final_mode = if layers.is_empty() {
            ShortcutMode::None
        } else {
            ShortcutMode::Add
        };
        let node_in = linear(parameter_tree, "node_in", head_config.model_dim, width, context).await?;
        let child_in = linear(parameter_tree, "child_in", head_config.model_dim + SLOT_NUMBERS, width, context).await?;
        let depth_embedding =
            parameter_tree.read_buffer("depth_embedding", &[head_config.max_depth + 1, width], DATA_TYPE).await?;
        let relation_bias =
            parameter_tree.read_buffer("relation_bias", &[config.num_heads, RELATIONS], DATA_TYPE).await?;
        let final_norm = norm(parameter_tree, "final_norm", width, final_mode, norm_config, context).await?;
        let couple_node = linear(parameter_tree, "couple_node", width, head_config.model_dim, context).await?;
        let couple_child = linear(parameter_tree, "couple_child", width, head_config.model_dim, context).await?;
        let head_dim = width / config.num_heads;
        if !matches!(head_dim, 32 | 64) {
            return Err(ConfidenceHeadError::UnsupportedShape(format!("tree attention heads of width {head_dim}")));
        }
        let attention =
            <B::Kernels as Kernels>::ConfidenceHeadTreeAttentionKernel::new(context, config.num_heads, head_dim)
                .map_err(ConfidenceHeadError::Backend)?;
        Ok(Self {
            config: config.clone(),
            node_in,
            child_in,
            depth_embedding,
            relation_bias,
            layers: layers.into_boxed_slice(),
            final_norm,
            couple_node,
            couple_child,
            activation: <B::Kernels as Kernels>::ActivationKernel::new(context, DATA_TYPE, true)
                .map_err(ConfidenceHeadError::Backend)?,
            input: <B::Kernels as Kernels>::ConfidenceHeadTreeInputKernel::new(context)
                .map_err(ConfidenceHeadError::Backend)?,
            relations: <B::Kernels as Kernels>::ConfidenceHeadTreeRelationsKernel::new(context)
                .map_err(ConfidenceHeadError::Backend)?,
            attention,
        })
    }

    pub fn encode(
        &self,
        node_inputs: B::ScratchBuffer,
        child_inputs: B::ScratchBuffer,
        frontier: impl BufferRef<Backend = B>,
        row_slots: &B::ScratchBuffer,
        slot_rows: &B::ScratchBuffer,
        slot_ancestors: impl BufferRef<Backend = B>,
        ancestor_stride: u32,
        slots: u32,
        rows: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(B::ScratchBuffer, B::ScratchBuffer), ConfidenceHeadError<B>> {
        if rows > TREE_ROWS_MAX {
            return Err(ConfidenceHeadError::UnsupportedShape(format!("tree attention over {rows} rows")));
        }
        let width = self.config.model_dim;
        let child_in = self.child_in.encode(child_inputs, rows - 1, command_buffer)?;
        let node_in = self.node_in.encode(node_inputs, slots, command_buffer)?;
        let mut hidden = command_buffer
            .allocate_scratch_for_shape(&[rows, width], DATA_TYPE)
            .map_err(ConfidenceHeadError::Backend)?;
        self.input.encode(
            &self.depth_embedding,
            &child_in,
            &node_in,
            frontier,
            row_slots,
            &mut hidden,
            rows,
            width,
            command_buffer,
        );

        let mut relations = command_buffer
            .allocate_scratch_for_shape(&[rows, rows], DataType::U32)
            .map_err(ConfidenceHeadError::Backend)?;
        self.relations.encode(
            frontier,
            slot_ancestors,
            slot_rows,
            &mut relations,
            rows,
            ancestor_stride,
            command_buffer,
        );
        let mut residual = command_buffer
            .allocate_scratch_for_shape(&[rows, width], DATA_TYPE)
            .map_err(ConfidenceHeadError::Backend)?;
        for layer in &self.layers {
            let normalized = layer
                .attention_norm
                .encode(&hidden, 0, rows, Some(&mut residual), command_buffer)
                .map_err(ConfidenceHeadError::Backend)?;
            let qkv = layer.qkv.encode(normalized, rows, command_buffer)?;
            let mut attended = command_buffer
                .allocate_scratch_for_shape(&[rows, width], DATA_TYPE)
                .map_err(ConfidenceHeadError::Backend)?;
            self.attention.encode(&qkv, &relations, &self.relation_bias, &mut attended, rows, command_buffer);
            let projected = layer.out.encode(attended, rows, command_buffer)?;
            let normalized = layer
                .mlp_norm
                .encode(&projected, 0, rows, Some(&mut residual), command_buffer)
                .map_err(ConfidenceHeadError::Backend)?;
            let mut raised = layer.up.encode(normalized, rows, command_buffer)?;
            self.activation.encode(
                None::<&B::ScratchBuffer>,
                &mut raised,
                rows * self.config.hidden_dim,
                ActivationType::SILU,
                command_buffer,
            );
            hidden = layer.down.encode(raised, rows, command_buffer)?;
        }
        let shortcut = (!self.layers.is_empty()).then_some(&mut residual);
        let finished =
            self.final_norm.encode(&hidden, 0, rows, shortcut, command_buffer).map_err(ConfidenceHeadError::Backend)?;
        let mut finished_copy =
            command_buffer.allocate_scratch(finished.size()).map_err(ConfidenceHeadError::Backend)?;
        command_buffer.encode_copy(&finished, &mut finished_copy);
        let node_shifts = self.couple_node.encode(finished, rows, command_buffer)?;
        let child_shifts = self.couple_child.encode(finished_copy, rows, command_buffer)?;
        Ok((node_shifts, child_shifts))
    }
}
