use proc_macros::uzu_config;

use crate::config::{
    linear::LinearConfig,
    mlp::{dense_mlp::DenseMLPConfig, routing_function::AnyRoutingFunction},
};

#[uzu_config(super::MLPConfig)]
pub struct MixtureOfExpertsConfig {
    pub expert_config: DenseMLPConfig,
    pub router_config: LinearConfig,
    pub routing_function: AnyRoutingFunction,

    pub num_routed_experts: usize,
    pub num_active_routed_experts: usize,
    pub router_has_biases: bool,

    pub num_shared_experts: usize,
    pub expert_hidden_dim: usize,
    pub gate_config: Option<LinearConfig>,
}
