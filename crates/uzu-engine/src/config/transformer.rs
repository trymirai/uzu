use uzu_engine_macros::uzu_config;

use crate::config::{
    normalization::NormalizationConfig, residual::AnyResidualConfig, transformer_layer::TransformerLayerConfig,
};

/// Layers `start_layer` to `end_layer - 1` run `num_passes` times with shared weights. Every pass restarts from the
/// stream that entered `start_layer`; every layer visit keeps its own state.
#[uzu_config]
pub struct LoopConfig {
    pub start_layer: u32,
    pub end_layer: u32,
    pub num_passes: u32,
}

/// One run of a layer: looped layers are visited once per pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerVisit {
    pub layer_index: u32,
    pub loop_pass: u32,
}

impl LoopConfig {
    pub fn is_looped(
        &self,
        layer_index: u32,
    ) -> bool {
        (self.start_layer..self.end_layer).contains(&layer_index)
    }

    pub fn layer_visits(
        &self,
        num_layers: u32,
    ) -> Box<[LayerVisit]> {
        let single = |layer_index| LayerVisit {
            layer_index,
            loop_pass: 0,
        };
        (0..self.start_layer)
            .map(single)
            .chain((0..self.num_passes).flat_map(|loop_pass| {
                (self.start_layer..self.end_layer).map(move |layer_index| LayerVisit {
                    layer_index,
                    loop_pass,
                })
            }))
            .chain((self.end_layer..num_layers).map(single))
            .collect()
    }
}

#[uzu_config]
pub struct TransformerConfig {
    pub residual_config: AnyResidualConfig,
    pub layer_configs: Box<[TransformerLayerConfig]>,
    pub output_norm_config: Option<NormalizationConfig>,
    pub model_dim: u32,
    pub hidden_dim: u32,
    #[serde(default)]
    pub loop_config: Option<LoopConfig>,
}

impl TransformerConfig {
    /// The layers in the order they run, one visit per looped pass.
    pub fn layer_visits(&self) -> Box<[LayerVisit]> {
        let num_layers = self.layer_configs.len() as u32;
        match &self.loop_config {
            Some(loop_config) => loop_config.layer_visits(num_layers),
            None => (0..num_layers)
                .map(|layer_index| LayerVisit {
                    layer_index,
                    loop_pass: 0,
                })
                .collect(),
        }
    }
}
