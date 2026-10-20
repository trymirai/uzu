use crate::{
    backends::Backend,
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
    encodable_block::{
        conf_head::{
            CONF_HEAD_SLOT_NUMBERS, ConfidenceHeadError,
            config::{ConfidenceHeadConfig, ConfidenceHeadTreeConfig},
        },
        linear::{Linear, LinearBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
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
        let norm_config = NormalizationConfig {
            epsilon: config.layer_norm_eps,
            scale_offset: None,
            upcast_mode: UpcastMode::FullLayer,
            subtract_mean: true,
            has_scale: true,
            has_biases: true,
        };

        let node_in = create_linear(context, &tree.subtree("node_in"), config.hidden, tree_config.width)?;
        let child_in = create_linear(
            context,
            &tree.subtree("child_in"),
            config.hidden + CONF_HEAD_SLOT_NUMBERS as u32,
            tree_config.width,
        )?;
        let layers = get_tree_layers(context, &tree, tree_config, &norm_config)?.into_boxed_slice();
        let final_norm =
            create_norm(context, &tree.subtree("final_norm"), tree_config.width, ShortcutMode::Add, &norm_config)?;
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

fn create_linear<B: Backend>(
    context: &B::Context,
    tree: &ParameterTree<B>,
    input_dim: u32,
    output_dim: u32,
) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
    <dyn Linear<B>>::new(input_dim, output_dim, DataType::F32, true, tree, context)
}

fn create_norm<B: Backend>(
    context: &B::Context,
    tree: &ParameterTree<B>,
    element_count: u32,
    shortcut_mode: ShortcutMode,
    config: &NormalizationConfig,
) -> Result<Normalization<B>, NormalizationNewError<B>> {
    Normalization::new(element_count, DataType::F32, None, shortcut_mode, PostLayerScalar::None, config, tree, context)
}

fn get_tree_layers<B: Backend>(
    context: &B::Context,
    tree: &ParameterTree<B>,
    tree_config: &ConfidenceHeadTreeConfig,
    norm_config: &NormalizationConfig,
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
            attention_norm: create_norm(
                context,
                &subtree.subtree("attention_norm"),
                tree_config.width,
                shortcut_mode,
                norm_config,
            )?,
            qkv: create_linear(context, &subtree.subtree("qkv"), tree_config.width, 3 * tree_config.width)?,
            out: create_linear(context, &subtree.subtree("out"), tree_config.width, tree_config.width)?,
            mlp_norm: create_norm(
                context,
                &subtree.subtree("mlp_norm"),
                tree_config.width,
                shortcut_mode,
                norm_config,
            )?,
            up: create_linear(context, &subtree.subtree("up"), tree_config.width, tree_config.mlp_hidden)?,
            down: create_linear(context, &subtree.subtree("down"), tree_config.mlp_hidden, tree_config.width)?,
        };
        tree_layers.push(layer);
    }

    Ok(tree_layers)
}
