use proc_macros::uzu_config;

use crate::config::dflash::DFlashDraftConfig;

#[allow(unused)] // TODO
#[uzu_config(super::ModelConfig)]
pub struct DFlashSpeculatorConfig {
    pub draft_config: DFlashDraftConfig,
}
