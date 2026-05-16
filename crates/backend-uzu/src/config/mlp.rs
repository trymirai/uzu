use proc_macros::uzu_config;

use crate::{backends::common::ActivationConfig, config::LinearConfig};

#[uzu_config]
#[serde(tag = "type")]
pub enum MLPConfig {
    #[serde(rename = "DenseMLPConfig")]
    Dense(DenseMLPConfig),
    #[serde(rename = "MixtureOfExpertsConfig")]
    MixtureOfExperts(MixtureOfExpertsConfig),
}

#[uzu_config]
pub struct DenseMLPConfig {
    pub linear_config: LinearConfig,
    pub activation: ActivationConfig,
    pub has_up_biases: bool,
    pub has_down_biases: bool,
    pub gate_clipping: Option<(Option<f32>, Option<f32>)>,
    pub up_clipping: Option<(Option<f32>, Option<f32>)>,
}

#[uzu_config]
pub struct MixtureOfExpertsConfig {
    pub expert_config: DenseMLPConfig,
    pub router_config: LinearConfig,
    pub routing_function: RoutingFunctionConfig,
    pub num_routed_experts: usize,
    pub num_active_routed_experts: usize,
    pub router_has_biases: bool,
    pub num_shared_experts: usize,
    pub expert_hidden_dim: usize,
    pub gate_config: Option<LinearConfig>,
}

#[uzu_config]
#[serde(tag = "type")]
pub enum RoutingFunctionConfig {
    #[serde(rename = "SoftmaxRouting")]
    SoftmaxRouting,
}
