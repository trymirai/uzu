use std::{
    fs::File,
    io::{self, BufReader},
    path::Path,
};

use thiserror::Error;

use crate::{
    backends::common::Backend,
    config::speculator::{AnySpeculatorConfig, model::SpeculatorModelConfig},
    data_type::DataType,
    encodable_block::{
        dflash::{DFlashDraft, DFlashDraftNewError},
        weaver::{Weaver, WeaverNewError},
    },
    engine::Engine,
    parameters::{HeaderLoadingError, ParameterLoader, ParameterLoaderError},
    speculators::dflash_speculator::DFlashSpeculator,
};

#[derive(Debug, Error)]
pub enum EngineLoadDFlashSpeculatorError<B: Backend> {
    #[error("I/O error: {0}")]
    IO(#[from] io::Error),
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("HeaderLoading error: {0}")]
    HeaderLoading(#[from] HeaderLoadingError),
    #[error("ParameterLoader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("DFlash draft error: {0}")]
    Draft(#[from] DFlashDraftNewError<B>),
    #[error("Weaver error: {0}")]
    Weaver(#[from] WeaverNewError<B>),
}

impl<B: Backend> Engine<B> {
    pub fn load_dflash_speculator(
        &self,
        model_path: &Path,
    ) -> Result<DFlashSpeculator<B>, EngineLoadDFlashSpeculatorError<B>> {
        let config: SpeculatorModelConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;
        let AnySpeculatorConfig::DFlashSpeculatorConfig(speculator_config) = config.speculator_config;

        let data_type = DataType::BF16;

        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weight_loader = ParameterLoader::new(&weights_file, &*self.context)?;
        let speculator_tree = weight_loader.tree().subtree("speculator")?;
        let draft_tree = speculator_tree.subtree("draft_model")?;
        let draft_model = DFlashDraft::new(&*self.context, &speculator_config.draft_config, &draft_tree, data_type)?;
        let weaver = speculator_config
            .weaver_config
            .as_ref()
            .map(|weaver_config| {
                Weaver::new(&*self.context, weaver_config, &speculator_tree.subtree("weaver")?, data_type)
            })
            .transpose()?;

        weight_loader.tree().assert_all_tensors_validated()?;

        Ok(DFlashSpeculator::new(draft_model, weaver, speculator_config))
    }
}
