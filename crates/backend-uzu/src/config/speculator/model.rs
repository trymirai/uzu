use proc_macros::uzu_config;

use crate::config::speculator::AnySpeculatorConfig;

#[uzu_config(super::SpeculatorConfig)]
pub struct SpeculatorModelConfig {
    pub speculator_config: AnySpeculatorConfig,
}
