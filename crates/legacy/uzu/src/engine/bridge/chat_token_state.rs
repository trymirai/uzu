use parking_lot::Mutex;
use shoji::traits::State as LlmInstanceState;
use uzu_engine::{backends::Backend, engine::language_model::state::LanguageModelState};

pub struct UzuChatTokenBackendInstanceState<B: Backend> {
    pub value: Mutex<LanguageModelState<B>>,
    pub sampling_seed: u64,
}

impl<B: Backend> UzuChatTokenBackendInstanceState<B> {
    pub fn new(
        value: LanguageModelState<B>,
        sampling_seed: u64,
    ) -> Self {
        Self {
            value: Mutex::new(value),
            sampling_seed,
        }
    }
}

impl<B: Backend> LlmInstanceState for UzuChatTokenBackendInstanceState<B> {}
