use uzu_engine_macros::uzu_config;

use crate::config::{AnyDraftAdapterConfig, confidence_head::ConfidenceHeadConfig, dflash::DFlashDraftConfig};

#[uzu_config(super::SpeculatorConfig)]
pub struct DFlashSpeculatorConfig {
    pub draft_config: DFlashDraftConfig,
    pub adapter_config: Option<AnyDraftAdapterConfig>,
    #[serde(default)]
    pub confidence_head_config: Option<ConfidenceHeadConfig>,
}
