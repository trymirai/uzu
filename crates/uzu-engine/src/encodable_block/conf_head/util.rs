use crate::{
    backends::Backend,
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
    encodable_block::{
        conf_head::ConfidenceHeadError,
        linear::{Linear, LinearBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
    },
    safetensors::ParameterTree,
};

pub fn create_linear<B: Backend>(
    context: &B::Context,
    tree: &ParameterTree<B>,
    input_dim: u32,
    output_dim: u32,
) -> Result<Box<dyn Linear<B>>, LinearBlockError<B>> {
    <dyn Linear<B>>::new(input_dim, output_dim, DataType::F32, true, tree, context)
}

pub fn create_normalization_config(epsilon: f32) -> NormalizationConfig {
    NormalizationConfig {
        epsilon,
        scale_offset: None,
        upcast_mode: UpcastMode::FullLayer,
        subtract_mean: true,
        has_scale: true,
        has_biases: true,
    }
}

pub fn create_normalization<B: Backend>(
    context: &B::Context,
    tree: &ParameterTree<B>,
    element_count: u32,
    shortcut_mode: ShortcutMode,
    epsilon: f32,
) -> Result<Normalization<B>, ConfidenceHeadError<B>> {
    let config = create_normalization_config(epsilon);
    Normalization::new(element_count, DataType::F32, None, shortcut_mode, PostLayerScalar::None, &config, tree, context)
        .map_err(map_norm_new_error)
}

pub fn map_norm_new_error<B: Backend>(err: NormalizationNewError<B>) -> ConfidenceHeadError<B> {
    match err {
        NormalizationNewError::Backend(error) => ConfidenceHeadError::Backend(error),
        NormalizationNewError::Parameter(error) => ConfidenceHeadError::Parameter(error),
    }
}
