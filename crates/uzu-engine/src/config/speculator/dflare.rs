use uzu_engine_macros::uzu_config;

use crate::config::{AnyDraftAdapterConfig, dflare::DFlareDraftConfig};

#[uzu_config(super::SpeculatorConfig)]
pub struct DFlareSpeculatorConfig {
    pub draft_config: DFlareDraftConfig,
    pub adapter_config: Option<AnyDraftAdapterConfig>,
}
