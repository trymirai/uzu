use proc_macros::uzu_config;

use crate::config::{activation::AnyActivation, linear::LinearConfig};

#[uzu_config(super::MLPConfig)]
pub struct DenseMLPConfig {
    pub linear_config: LinearConfig,
    pub activation: AnyActivation,
    pub has_up_biases: bool,
    pub has_down_biases: bool,
    pub gate_clipping: Option<(Option<f32>, Option<f32>)>,
    pub up_clipping: Option<(Option<f32>, Option<f32>)>,
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
            gate_clipping: None,
            up_clipping: None,
        }
    }
}
