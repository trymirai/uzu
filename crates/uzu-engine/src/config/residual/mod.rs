use uzu_engine_macros::uzu_config_abstract;

pub mod additive;
pub mod gated;

#[uzu_config_abstract(additive::AdditiveResidualConfig, gated::GatedResidualConfig)]
pub struct ResidualConfig;
