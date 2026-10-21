use crate::{
    backends::Backend,
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

pub struct ConfidenceHeadTree<B: Backend> {
    node_in: Box<dyn Linear<B>>,
    child_in: Box<dyn Linear<B>>,
    layers: Box<[ConfidenceHeadTreeLayer<B>]>,
    final_norm: Normalization<B>,
    couple_node: Box<dyn Linear<B>>,
    couple_child: Box<dyn Linear<B>>,
    depth_embedding: B::GlobalBuffer,
    relation_weights: B::GlobalBuffer,
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
            ShortcutMode::Add,
            config.layer_norm_eps,
        )?;
        let couple_node = create_linear(context, &tree.subtree("couple_node"), tree_config.width, config.hidden)?;
        let couple_child = create_linear(context, &tree.subtree("couple_child"), tree_config.width, config.hidden)?;
        let depth_embedding =
            tree.read_buffer("depth_embedding", &[config.max_depth + 1, tree_config.width], DataType::F32)?;
        let relation_weights =
            tree.read_buffer("relation_weight", &[tree_config.heads, tree_config.relations], DataType::F32)?;

        Ok(Self {
            node_in,
            child_in,
            layers,
            final_norm,
            couple_node,
            couple_child,
            depth_embedding,
            relation_weights,
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
                shortcut_mode,
                config.layer_norm_eps,
            )?,
            up: create_linear(context, &subtree.subtree("up"), tree_config.width, tree_config.mlp_hidden)?,
            down: create_linear(context, &subtree.subtree("down"), tree_config.mlp_hidden, tree_config.width)?,
        };
        tree_layers.push(layer);
    }

    Ok(tree_layers)
}
