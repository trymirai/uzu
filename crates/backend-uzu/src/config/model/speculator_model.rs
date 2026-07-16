use proc_macros::uzu_config;

use crate::config::speculator::AnySpeculatorConfig;

#[uzu_config(super::BaseModelConfig)]
pub struct SpeculatorModelConfig {
    pub speculator_config: AnySpeculatorConfig,
}
