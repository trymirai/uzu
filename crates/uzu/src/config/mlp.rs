use serde::{Deserialize, Serialize};

use crate::{backends::common::ActivationConfig, config::LinearConfig};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum MLPConfig {
    #[serde(rename = "DenseMLPConfig")]
    Dense(DenseMLPConfig),
    #[serde(rename = "MixtureOfExpertsConfig")]
    MixtureOfExperts(MixtureOfExpertsConfig),
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DenseMLPConfig {
    pub linear_config: LinearConfig,
    pub activation: ActivationConfig,
    #[serde(default)]
    pub has_up_biases: bool,
    #[serde(default)]
    pub has_down_biases: bool,
    #[serde(default)]
    pub gate_clipping: Option<[Option<f32>; 2]>,
    #[serde(default)]
    pub up_clipping: Option<[Option<f32>; 2]>,
    #[serde(default = "default_activation_to_gate")]
    pub activation_to_gate: bool,
}

fn default_activation_to_gate() -> bool {
    true
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct MixtureOfExpertsConfig {
    pub num_routed_experts: usize,
    pub num_active_routed_experts: usize,
    pub routing_function: RoutingFunctionConfig,
    pub router_config: LinearConfig,
    pub router_has_biases: bool,
    pub expert_config: MoeExpertConfig,
    #[serde(default)]
    pub num_shared_experts: Option<usize>,
    #[serde(default)]
    pub expert_hidden_dim: Option<usize>,
    #[serde(default)]
    pub gate_config: Option<LinearConfig>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum RoutingFunctionConfig {
    #[serde(rename = "SoftmaxRouting")]
    SoftmaxRouting,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct MoeExpertConfig {
    pub linear_config: LinearConfig,
    pub activation: ActivationConfig,
    pub has_up_biases: bool,
    pub has_down_biases: bool,
    #[serde(default)]
    pub gate_clipping: Option<[Option<f32>; 2]>,
    #[serde(default)]
    pub up_clipping: Option<[Option<f32>; 2]>,
}

#[cfg(test)]
#[path = "../../tests/unit/config/mlp_test.rs"]
mod tests;
