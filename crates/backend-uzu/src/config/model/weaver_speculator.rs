use proc_macros::uzu_config;

use crate::config::{dflash::DFlashDraftConfig, weaver::WeaverConfig};

#[allow(unused)] // TODO: remove once Weaver traversal wiring consumes the config.
#[uzu_config(super::ModelConfig)]
pub struct WeaverSpeculatorConfig {
    pub draft_config: DFlashDraftConfig,
    pub weaver_config: WeaverConfig,
    pub tree_budget: usize,
}
