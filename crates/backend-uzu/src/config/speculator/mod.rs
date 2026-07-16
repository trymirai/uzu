use proc_macros::uzu_config_abstract;

pub mod dflash;

#[uzu_config_abstract(dflash::DFlashSpeculatorConfig)]
pub struct SpeculatorConfig;
