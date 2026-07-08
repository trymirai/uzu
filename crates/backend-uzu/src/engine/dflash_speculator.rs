use std::{
    fs::File,
    io::{self, BufReader},
    path::Path,
};

use thiserror::Error;

use crate::{
    backends::common::Backend,
    config::model::dflash_speculator::DFlashSpeculatorConfig,
    data_type::DataType,
    encodable_block::dflash::{DFlashDraft, DFlashDraftNewError},
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
}

impl<B: Backend> Engine<B> {
    pub fn load_dflash_speculator(
        &self,
        model_path: &Path,
    ) -> Result<DFlashSpeculator<B>, EngineLoadDFlashSpeculatorError<B>> {
        let context = self.context.as_ref();

        let config: DFlashSpeculatorConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;

        let data_type = DataType::BF16;

        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weight_loader = ParameterLoader::new(&weights_file, context)?;
        let draft_tree = weight_loader.tree().subtree("draft_model")?;
        let draft_model = DFlashDraft::new(&*self.context, &config.draft_config, &draft_tree, data_type)?;

        weight_loader.tree().assert_all_tensors_validated()?;

        Ok(DFlashSpeculator::new(draft_model, config))
    }
}
