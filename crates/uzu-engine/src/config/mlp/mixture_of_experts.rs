use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig,
    mlp::{dense_mlp::DenseMLPConfig, routing_function::AnyRoutingFunction},
};

#[uzu_config(super::MLPConfig)]
pub struct MixtureOfExpertsConfig {
    pub expert_config: DenseMLPConfig,
    pub router_config: LinearConfig,
    pub routing_function: AnyRoutingFunction,

    pub num_routed_experts: u32,
    pub num_active_routed_experts: u32,
    pub router_has_biases: bool,

    pub num_shared_experts: u32,
    pub expert_hidden_dim: u32,
    pub gate_config: Option<LinearConfig>,
}
