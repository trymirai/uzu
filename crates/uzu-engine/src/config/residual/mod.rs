use uzu_engine_macros::uzu_config_abstract;

pub mod additive;
pub mod block_attention;
pub mod gated;

#[uzu_config_abstract(
    additive::AdditiveResidualConfig,
    gated::GatedResidualConfig,
    block_attention::BlockAttentionResidualConfig
)]
pub struct ResidualConfig;
