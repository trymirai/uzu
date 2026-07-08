use crate::{
    backends::common::Backend,
    config::model::weaver_speculator::WeaverSpeculatorConfig,
    encodable_block::{dflash::DFlashDraft, weaver::Weaver},
};

#[allow(dead_code)] // TODO: remove once Weaver traversal wiring consumes the loaded speculator.
pub struct WeaverSpeculator<B: Backend> {
    pub(crate) draft_model: DFlashDraft<B>,
    pub(crate) weaver: Weaver<B>,
    pub(crate) config: WeaverSpeculatorConfig,
}

impl<B: Backend> WeaverSpeculator<B> {
    pub(crate) fn new(
        draft_model: DFlashDraft<B>,
        weaver: Weaver<B>,
        config: WeaverSpeculatorConfig,
    ) -> Self {
        Self {
            draft_model,
            weaver,
            config,
        }
    }
}
