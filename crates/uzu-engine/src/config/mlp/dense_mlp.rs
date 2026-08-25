use uzu_engine_macros::uzu_config;

use crate::config::{activation::AnyActivation, clipping::ClippingBounds, linear::LinearConfig};

#[uzu_config(super::MLPConfig)]
pub struct DenseMLPConfig {
    pub linear_config: LinearConfig,
    pub activation: AnyActivation,
    pub has_up_biases: bool,
    pub has_down_biases: bool,
    pub gate_clipping: ClippingBounds,
    pub up_clipping: ClippingBounds,
}

impl DenseMLPConfig {
    pub fn unclipped(
        linear_config: LinearConfig,
        activation: AnyActivation,
        has_up_biases: bool,
        has_down_biases: bool,
    ) -> Self {
        Self {
            ty: Default::default(),
            linear_config,
            activation,
            has_up_biases,
            has_down_biases,
            gate_clipping: ClippingBounds::default(),
            up_clipping: ClippingBounds::default(),
        }
    }
}
