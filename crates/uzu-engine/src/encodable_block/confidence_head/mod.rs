mod edge_weights;
mod error;
mod head;
mod node_weights;
mod tree_attention;

pub use error::ConfidenceHeadError;
pub use head::{ConfidenceHead, ConfidenceHeadStep};
use tree_attention::TreeAttention;

use crate::{
    backends::common::Backend,
    config::normalization::NormalizationConfig,
    data_type::DataType,
    encodable_block::{
        linear::{Linear, LinearBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
    },
    safetensors::ParameterTree,
};

const DATA_TYPE: DataType = DataType::BF16;

async fn linear<B: Backend>(
    parameter_tree: &ParameterTree<'_, B>,
    name: &str,
    inputs: u32,
    outputs: u32,
    context: &B::Context,
) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
    <dyn Linear<B>>::new(inputs, outputs, DATA_TYPE, true, &parameter_tree.subtree(name), context).await
}

async fn norm<B: Backend>(
    parameter_tree: &ParameterTree<'_, B>,
    name: &str,
    element_count: u32,
    shortcut_mode: ShortcutMode,
    config: &NormalizationConfig,
    context: &B::Context,
) -> Result<Normalization<B>, NormalizationNewError<B>> {
    Normalization::new(
        element_count,
        DATA_TYPE,
        DATA_TYPE,
        None,
        shortcut_mode,
        PostLayerScalar::None,
        config,
        &parameter_tree.subtree(name),
        context,
    )
    .await
}
