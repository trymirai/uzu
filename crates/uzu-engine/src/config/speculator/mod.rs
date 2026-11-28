use uzu_engine_macros::uzu_config_abstract;

pub mod dflare;
pub mod dflash;

#[uzu_config_abstract(dflash::DFlashSpeculatorConfig, dflare::DFlareSpeculatorConfig)]
pub struct SpeculatorConfig;
