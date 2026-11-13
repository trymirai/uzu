use uzu_engine_macros::uzu_config_abstract;

pub mod dense_mlp;
pub mod mixture_of_experts;
pub mod routing_function;
pub mod ungated;

#[uzu_config_abstract(dense_mlp::DenseMLPConfig, mixture_of_experts::MixtureOfExpertsConfig, ungated::UngatedMLPConfig)]
pub struct MLPConfig;
