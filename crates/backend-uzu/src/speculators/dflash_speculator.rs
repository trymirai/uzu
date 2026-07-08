use crate::{
    backends::common::Backend, config::model::dflash_speculator::DFlashSpeculatorConfig,
    encodable_block::dflash::DFlashDraft,
};

// TODO: remove once traversal verification consumes the loaded DFlash model.
#[allow(dead_code)]
pub struct DFlashSpeculator<B: Backend> {
    pub(crate) draft: DFlashDraft<B>,
    pub(crate) config: DFlashSpeculatorConfig,
}

impl<B: Backend> DFlashSpeculator<B> {
    pub(crate) fn new(
        draft: DFlashDraft<B>,
        config: DFlashSpeculatorConfig,
    ) -> Self {
        Self {
            draft,
            config,
        }
    }
}
