use crate::{
    backends::common::Backend,
    config::speculator::dflash::DFlashSpeculatorConfig,
    encodable_block::{dflash::DFlashDraft, weaver::Weaver},
};

#[allow(dead_code)]
pub struct DFlashSpeculator<B: Backend> {
    pub(crate) model: DFlashDraft<B>,
    pub(crate) weaver: Option<Weaver<B>>,
    pub(crate) config: DFlashSpeculatorConfig,
}

impl<B: Backend> DFlashSpeculator<B> {
    pub(crate) const fn new(
        model: DFlashDraft<B>,
        weaver: Option<Weaver<B>>,
        config: DFlashSpeculatorConfig,
    ) -> Self {
        Self {
            model,
            weaver,
            config,
        }
    }
}
