use crate::{
    backends::common::{Backend, gpu_types::qkv_norm::QKVNormParams},
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
    safetensors::{ParameterLoaderError, ParameterTree},
};

pub struct QKVNorm<B: Backend> {
    pub params: QKVNormParams,
    pub scales: [Option<B::GlobalGpuBuffer>; 3],
}

impl<B: Backend> QKVNorm<B> {
    pub async fn new(
        head_dim: u32,
        query: Option<NormalizationConfig>,
        key: Option<NormalizationConfig>,
        value: Option<NormalizationConfig>,
        parameter_tree: &ParameterTree<'_, B>,
    ) -> Result<Self, ParameterLoaderError<B>> {
        let mut params = QKVNormParams::default();
        let mut scales = [None, None, None];
        for (i, (config, name)) in
            [(query, "query_norm"), (key, "key_norm"), (value, "value_norm")].into_iter().enumerate()
        {
            let Some(config) = config else {
                continue;
            };
            params.enabled[i] = 1;
            params.full_layer[i] = u32::from(config.upcast_mode == UpcastMode::FullLayer);
            params.epsilon[i] = config.epsilon;
            params.scale_offset[i] = config.scale_offset.unwrap_or(0.0);
            if config.has_scale {
                scales[i] = Some(parameter_tree.subtree(name).read_buffer("scales", &[head_dim], DataType::F32).await?);
            }
        }
        Ok(Self {
            params,
            scales,
        })
    }
}
