use crate::{
    backends::{
        Backend,
        common::{
            Buffer, BufferRef, CommandBuffer, CommandBufferEncoding, Kernels,
            gpu_types::activation_type::ActivationType,
            kernel::{ActivationKernel, ConfidenceHeadTreeAttentionKernel, ConfidenceHeadTreeTokensKernel},
        },
    },
    data_type::DataType,
    encodable_block::{
        conf_head::{
            CONF_HEAD_SLOT_NUMBERS, ConfidenceHeadError,
            config::{ConfidenceHeadConfig, ConfidenceHeadTreeConfig},
            util::{create_linear, create_normalization},
        },
        linear::Linear,
        normalization::{Normalization, ShortcutMode},
    },
    safetensors::ParameterTree,
};

pub struct ConfidenceHeadTreeLayer<B: Backend> {
    attention_norm: Normalization<B>,
    qkv: Box<dyn Linear<B>>,
    out: Box<dyn Linear<B>>,
    mlp_norm: Normalization<B>,
    up: Box<dyn Linear<B>>,
    down: Box<dyn Linear<B>>,
}

pub(super) struct ConfidenceHeadTreeOutput<B: Backend> {
    pub node_shifts: B::ScratchBuffer,
    pub child_shifts: B::ScratchBuffer,
    pub slot_tokens: B::ScratchBuffer,
    pub token_slots: B::ScratchBuffer,
}

pub struct ConfidenceHeadTree<B: Backend> {
    config: ConfidenceHeadTreeConfig,
    max_depth: u32,
    node_in: Box<dyn Linear<B>>,
    child_in: Box<dyn Linear<B>>,
    layers: Box<[ConfidenceHeadTreeLayer<B>]>,
    final_norm: Normalization<B>,
    couple_node: Box<dyn Linear<B>>,
    couple_child: Box<dyn Linear<B>>,
    depth_embedding: B::GlobalBuffer,
    relation_weights: B::GlobalBuffer,
    tree_tokens: <B::Kernels as Kernels>::ConfidenceHeadTreeTokensKernel,
    tree_attention: <B::Kernels as Kernels>::ConfidenceHeadTreeAttentionKernel,
    activation: <B::Kernels as Kernels>::ActivationKernel,
}

impl<B: Backend> ConfidenceHeadTree<B> {
    pub fn new(
        context: &B::Context,
        parameters: &ParameterTree<B>,
        config: &ConfidenceHeadConfig,
        tree_config: &ConfidenceHeadTreeConfig,
    ) -> Result<Self, ConfidenceHeadError<B>> {
        let tree = parameters.subtree("tree");

        let node_in = create_linear(context, &tree.subtree("node_in"), config.hidden, tree_config.width)?;
        let child_in = create_linear(
            context,
            &tree.subtree("child_in"),
            config.hidden + CONF_HEAD_SLOT_NUMBERS as u32,
            tree_config.width,
        )?;
        let layers = get_tree_layers(context, &tree, config, tree_config)?.into_boxed_slice();
        let final_norm = create_normalization(
            context,
            &tree.subtree("final_norm"),
            tree_config.width,
            if tree_config.layers == 0 {
                ShortcutMode::None
            } else {
                ShortcutMode::Add
            },
            config.layer_norm_eps,
        )?;
        let couple_node = create_linear(context, &tree.subtree("couple_node"), tree_config.width, config.hidden)?;
        let couple_child = create_linear(context, &tree.subtree("couple_child"), tree_config.width, config.hidden)?;
        let depth_embedding =
            tree.read_buffer("depth_embedding", &[config.max_depth + 1, tree_config.width], DataType::F32)?;
        let relation_weights =
            tree.read_buffer("relation_weight", &[tree_config.heads, tree_config.relations], DataType::F32)?;

        Ok(Self {
            config: tree_config.clone(),
            max_depth: config.max_depth,
            node_in,
            child_in,
            layers,
            final_norm,
            couple_node,
            couple_child,
            depth_embedding,
            relation_weights,
            tree_tokens: <B::Kernels as Kernels>::ConfidenceHeadTreeTokensKernel::new(context)
                .map_err(ConfidenceHeadError::Backend)?,
            tree_attention: <B::Kernels as Kernels>::ConfidenceHeadTreeAttentionKernel::new(
                context,
                tree_config.heads,
                tree_config.width / tree_config.heads,
                tree_config.relations,
                config.max_depth,
            )
            .map_err(ConfidenceHeadError::Backend)?,
            activation: <B::Kernels as Kernels>::ActivationKernel::new(context, DataType::F32, true)
                .map_err(ConfidenceHeadError::Backend)?,
        })
    }

    /// Refines the grown tree's node and child features using relation-biased attention.
    pub(super) fn encode(
        &self,
        nodes: &B::ScratchBuffer,
        child_inputs: &B::ScratchBuffer,
        child_real: &B::ScratchBuffer,
        packed_tree: impl BufferRef<Backend = B>,
        frontier: impl BufferRef<Backend = B>,
        slots: u32,
        children: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<ConfidenceHeadTreeOutput<B>, ConfidenceHeadError<B>> {
        let token_count = 1 + slots * children;
        let width = self.config.width;

        let mut nodes_copy = command_buffer.allocate_scratch(nodes.size()).map_err(ConfidenceHeadError::Backend)?;
        command_buffer.encode_copy(nodes, &mut nodes_copy);
        let node_inputs = self.node_in.encode(nodes_copy, slots, command_buffer)?;

        let mut children_copy =
            command_buffer.allocate_scratch(child_inputs.size()).map_err(ConfidenceHeadError::Backend)?;
        command_buffer.encode_copy(child_inputs, &mut children_copy);
        let child_inputs = self.child_in.encode(children_copy, slots * children, command_buffer)?;

        let mut tokens = command_buffer
            .allocate_scratch_for_shape(&[token_count, width], DataType::F32)
            .map_err(ConfidenceHeadError::Backend)?;
        let mut token_parents = command_buffer
            .allocate_scratch_for_shape(&[token_count], DataType::U32)
            .map_err(ConfidenceHeadError::Backend)?;
        let mut token_depths = command_buffer
            .allocate_scratch_for_shape(&[token_count], DataType::U32)
            .map_err(ConfidenceHeadError::Backend)?;
        let mut token_present = command_buffer
            .allocate_scratch_for_shape(&[token_count], DataType::U32)
            .map_err(ConfidenceHeadError::Backend)?;
        let mut token_slots = command_buffer
            .allocate_scratch_for_shape(&[token_count], DataType::U32)
            .map_err(ConfidenceHeadError::Backend)?;
        let mut token_ancestors = command_buffer
            .allocate_scratch_for_shape(&[token_count, self.max_depth + 1], DataType::U32)
            .map_err(ConfidenceHeadError::Backend)?;
        let mut slot_tokens =
            command_buffer.allocate_scratch_for_shape(&[slots], DataType::U32).map_err(ConfidenceHeadError::Backend)?;
        command_buffer.encode_fill(&mut slot_tokens, 0);

        self.tree_tokens.encode(
            packed_tree,
            frontier,
            child_real,
            &node_inputs,
            &child_inputs,
            &self.depth_embedding,
            &mut tokens,
            &mut token_parents,
            &mut token_depths,
            &mut token_present,
            &mut token_slots,
            &mut token_ancestors,
            &mut slot_tokens,
            slots,
            children,
            width,
            self.max_depth,
            command_buffer,
        );

        let mut residual = command_buffer
            .allocate_scratch_for_shape(&[token_count, width], DataType::F32)
            .map_err(ConfidenceHeadError::Backend)?;
        let mut layer_input = tokens;
        for layer in &self.layers {
            let attention_input = layer
                .attention_norm
                .encode(&layer_input, 0, token_count, Some(&mut residual), command_buffer)
                .map_err(ConfidenceHeadError::Backend)?;
            let qkv = layer.qkv.encode(attention_input, token_count, command_buffer)?;
            let mut attention = command_buffer
                .allocate_scratch_for_shape(&[token_count, width], DataType::F32)
                .map_err(ConfidenceHeadError::Backend)?;
            self.tree_attention.encode(
                &qkv,
                &self.relation_weights,
                &token_parents,
                &token_depths,
                &token_present,
                &token_ancestors,
                &mut attention,
                token_count,
                command_buffer,
            );
            let projected = layer.out.encode(attention, token_count, command_buffer)?;
            let mlp_input = layer
                .mlp_norm
                .encode(&projected, 0, token_count, Some(&mut residual), command_buffer)
                .map_err(ConfidenceHeadError::Backend)?;
            let mut raised = layer.up.encode(mlp_input, token_count, command_buffer)?;
            self.activation.encode(
                None::<&B::ScratchBuffer>,
                &mut raised,
                token_count * self.config.mlp_hidden,
                ActivationType::SILU,
                command_buffer,
            );
            layer_input = layer.down.encode(raised, token_count, command_buffer)?;
        }

        let shortcut = if self.layers.is_empty() {
            None
        } else {
            Some(&mut residual)
        };
        let normed = self
            .final_norm
            .encode(&layer_input, 0, token_count, shortcut, command_buffer)
            .map_err(ConfidenceHeadError::Backend)?;
        let mut normed_copy = command_buffer.allocate_scratch(normed.size()).map_err(ConfidenceHeadError::Backend)?;
        command_buffer.encode_copy(&normed, &mut normed_copy);

        Ok(ConfidenceHeadTreeOutput {
            node_shifts: self.couple_node.encode(normed, token_count, command_buffer)?,
            child_shifts: self.couple_child.encode(normed_copy, token_count, command_buffer)?,
            slot_tokens,
            token_slots,
        })
    }
}

fn get_tree_layers<B: Backend>(
    context: &B::Context,
    tree: &ParameterTree<B>,
    config: &ConfidenceHeadConfig,
    tree_config: &ConfidenceHeadTreeConfig,
) -> Result<Vec<ConfidenceHeadTreeLayer<B>>, ConfidenceHeadError<B>> {
    let mut tree_layers: Vec<ConfidenceHeadTreeLayer<B>> = Vec::new();
    for i in 0..tree_config.layers {
        let subtree = tree.subtree("layers").subtree(&i.to_string());
        let shortcut_mode = if i == 0 {
            ShortcutMode::Copy
        } else {
            ShortcutMode::Add
        };
        let layer = ConfidenceHeadTreeLayer {
            attention_norm: create_normalization(
                context,
                &subtree.subtree("attention_norm"),
                tree_config.width,
                shortcut_mode,
                config.layer_norm_eps,
            )?,
            qkv: create_linear(context, &subtree.subtree("qkv"), tree_config.width, 3 * tree_config.width)?,
            out: create_linear(context, &subtree.subtree("out"), tree_config.width, tree_config.width)?,
            mlp_norm: create_normalization(
                context,
                &subtree.subtree("mlp_norm"),
                tree_config.width,
                ShortcutMode::Add,
                config.layer_norm_eps,
            )?,
            up: create_linear(context, &subtree.subtree("up"), tree_config.width, tree_config.mlp_hidden)?,
            down: create_linear(context, &subtree.subtree("down"), tree_config.mlp_hidden, tree_config.width)?,
        };
        tree_layers.push(layer);
    }

    Ok(tree_layers)
}

#[cfg(test)]
#[path = "../../../unit/encodable_block/confidence_head_tree_test.rs"]
mod tests;
