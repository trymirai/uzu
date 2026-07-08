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
};

#[allow(dead_code)]
pub struct DFlashSpeculator<B: Backend> {
    draft: DFlashDraft<B>,
    block_size: usize,
    mask_token_id: u64,
    target_layer_ids: Box<[usize]>,
    vocab_size: usize,
}

#[allow(dead_code)]
impl<B: Backend> DFlashSpeculator<B> {
    pub(crate) fn draft(&self) -> &DFlashDraft<B> {
        &self.draft
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(crate) fn mask_token_id(&self) -> u64 {
        self.mask_token_id
    }

    pub(crate) fn target_layer_ids(&self) -> &[usize] {
        &self.target_layer_ids
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub(crate) fn max_context_length(&self) -> Option<usize> {
        self.draft.max_context_length()
    }
}

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
    #[error("DFlash speculator config error: {0}")]
    Config(#[from] DFlashSpeculatorConfigError),
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DFlashSpeculatorConfigError {
    #[error("DFlash block_size must include root plus at least one draft token")]
    InvalidBlockSize,
    #[error("invalid DFlash target_layer_ids")]
    InvalidTargetLayerIds,
    #[error("DFlash draft must contain at least one layer")]
    EmptyDraft,
}

fn validate_dflash_speculator_config(config: &DFlashSpeculatorConfig) -> Result<(), DFlashSpeculatorConfigError> {
    if config.draft_config.block_size <= 1 {
        return Err(DFlashSpeculatorConfigError::InvalidBlockSize);
    }
    if config.draft_config.target_layer_ids.is_empty()
        || config
            .draft_config
            .target_layer_ids
            .iter()
            .any(|target_layer_id| *target_layer_id >= config.draft_config.num_target_layers)
        || config
            .draft_config
            .target_layer_ids
            .iter()
            .enumerate()
            .any(|(index, target_layer_id)| config.draft_config.target_layer_ids[..index].contains(target_layer_id))
    {
        return Err(DFlashSpeculatorConfigError::InvalidTargetLayerIds);
    }
    if config.draft_config.layer_configs.is_empty() {
        return Err(DFlashSpeculatorConfigError::EmptyDraft);
    }
    Ok(())
}

impl<B: Backend> Engine<B> {
    pub fn load_dflash_speculator(
        &self,
        model_path: &Path,
    ) -> Result<DFlashSpeculator<B>, EngineLoadDFlashSpeculatorError<B>> {
        let config: DFlashSpeculatorConfig =
            serde_json::from_reader(BufReader::new(File::open(model_path.join("config.json"))?))?;
        validate_dflash_speculator_config(&config)?;

        let weights_file = File::open(model_path.join("model.safetensors"))?;
        let weight_loader = ParameterLoader::new(&weights_file, self.context.as_ref())?;
        let draft_tree = weight_loader.tree().subtree("draft_model")?;
        let draft = DFlashDraft::new(self.context.as_ref(), &config.draft_config, &draft_tree, DataType::BF16)?;

        weight_loader.tree().assert_all_tensors_validated()?;

        Ok(DFlashSpeculator {
            draft,
            block_size: config.draft_config.block_size,
            mask_token_id: config.draft_config.mask_token_id,
            target_layer_ids: config.draft_config.target_layer_ids,
            vocab_size: config.draft_config.vocab_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use proc_macros::uzu_test;

    use crate::{backends::cpu::Cpu, engine::Engine};

    const DFLASH_PATH_ENV: &str = "UZU_DFLASH_LOAD_SMOKE";
    const DEFAULT_DFLASH_PATH: &str = "/private/tmp/lalamo-mini-dflash";

    fn smoke_path() -> Option<PathBuf> {
        let path =
            std::env::var_os(DFLASH_PATH_ENV).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(DEFAULT_DFLASH_PATH));
        path.exists().then_some(path)
    }

    #[uzu_test]
    #[ignore]
    fn test_dflash_load_smoke() {
        let Some(path) = smoke_path() else {
            return;
        };
        let engine = Engine::<Cpu>::new().expect("engine");
        let dflash = engine.load_dflash_speculator(&path).expect("DFlash load");

        assert!(dflash.block_size() > 1);
        assert!(dflash.mask_token_id() < dflash.vocab_size() as u64);
        assert!(!dflash.target_layer_ids().is_empty());
        assert!(dflash.draft().model_dim() > 0);
    }
}
