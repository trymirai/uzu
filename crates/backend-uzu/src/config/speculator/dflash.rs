use proc_macros::uzu_config;

use crate::config::{dflash::DFlashDraftConfig, weaver::WeaverConfig};

#[uzu_config(super::SpeculatorConfig)]
pub struct DFlashSpeculatorConfig {
    pub draft_config: DFlashDraftConfig,
    pub weaver_config: Option<WeaverConfig>,
}
