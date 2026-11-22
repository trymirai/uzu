use uzu_engine_macros::uzu_config_abstract;

pub mod dflash;
pub mod dspark;

#[uzu_config_abstract(dflash::DFlashSpeculatorConfig, dspark::DSparkSpeculatorConfig)]
pub struct SpeculatorConfig;
