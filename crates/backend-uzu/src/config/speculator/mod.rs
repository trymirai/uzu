use proc_macros::uzu_config_abstract;

pub mod dflash;
pub mod model;

#[uzu_config_abstract(dflash::DFlashSpeculatorConfig)]
pub struct SpeculatorConfig;
