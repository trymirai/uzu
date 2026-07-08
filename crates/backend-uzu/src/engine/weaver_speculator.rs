use std::{
    fs::File,
    io::{self, BufReader},
    path::Path,
};

use thiserror::Error;

use crate::{
    backends::common::Backend,
    config::model::weaver_speculator::WeaverSpeculatorConfig,
    data_type::DataType,
    encodable_block::{
        dflash::{DFlashDraft, DFlashDraftNewError},
        weaver::{Weaver, WeaverNewError},
    },
    engine::Engine,
    parameters::{HeaderLoadingError, ParameterLoader, ParameterLoaderError},
    speculators::weaver_speculator::WeaverSpeculator,
};

#[derive(Debug, Error)]
pub enum EngineLoadWeaverSpeculatorError<B: Backend> {
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
    pub fn load_weaver_speculator(
        &self,
        model_path: &Path,
    ) -> Result<WeaverSpeculator<B>, EngineLoadWeaverSpeculatorError<B>> {
        let config: WeaverSpeculatorConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;

        let draft_data_type = DataType::BF16;
        let weaver_data_type = DataType::F32;

        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weight_loader = ParameterLoader::new(&weights_file, &*self.context)?;
        let draft_tree = weight_loader.tree().subtree("draft_model")?;
        let draft_model = DFlashDraft::new(&*self.context, &config.draft_config, &draft_tree, draft_data_type)?;
        let weaver_tree = weight_loader.tree().subtree("weaver")?;
        let weaver = Weaver::new(&*self.context, &config.weaver_config, &weaver_tree, weaver_data_type)?;

        weight_loader.tree().assert_all_tensors_validated()?;

        Ok(WeaverSpeculator::new(draft_model, weaver, config))
    }
}
