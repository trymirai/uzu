use std::sync::Arc;

use parking_lot::Mutex;
use shoji::traits::State as LlmInstanceState;

use crate::{backends::common::Backend, engine::language_model::state::LanguageModelState};

pub struct UzuChatTokenBackendInstanceState<B: Backend> {
    pub value: Arc<Mutex<LanguageModelState<B>>>,
}

impl<B: Backend> UzuChatTokenBackendInstanceState<B> {
    pub fn new(value: LanguageModelState<B>) -> Self {
        Self {
            value: Arc::new(Mutex::new(value)),
        }
    }
}

impl<B: Backend> LlmInstanceState for UzuChatTokenBackendInstanceState<B> {}
