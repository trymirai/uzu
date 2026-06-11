// TODO: remove when implementing a consumer (expect warns once this is used).
#![expect(dead_code)]

use std::{fs::File, io::BufReader, path::Path};

use proc_macros::uzu_config;

use crate::{config::dflash::DFlashDraftConfig, session::types::Error};

#[uzu_config(super::ModelConfig)]
pub struct DFlashSpeculatorConfig {
    pub draft_config: DFlashDraftConfig,
}

impl DFlashSpeculatorConfig {
    pub fn new(speculator_path: &Path) -> Result<DFlashSpeculatorConfig, Error> {
        let config_path = speculator_path.join("config.json");
        let config_file = File::open(&config_path)?;
        let config: DFlashSpeculatorConfig = serde_json::from_reader(BufReader::new(config_file))?;
        Ok(config)
    }
}
