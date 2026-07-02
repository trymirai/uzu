use shoji::traits::State as LlmInstanceState;

use crate::{
    backends::common::Backend, bridge::sync_shared::SyncShared, engine::language_model::state::LanguageModelState,
};

pub struct UzuChatTokenBackendInstanceState<B: Backend> {
    pub value: SyncShared<LanguageModelState<B>>,
}

impl<B: Backend> UzuChatTokenBackendInstanceState<B> {
    pub fn new(value: LanguageModelState<B>) -> Self {
        Self {
            value: SyncShared::new(value),
        }
    }
}

impl<B: Backend> LlmInstanceState for UzuChatTokenBackendInstanceState<B> {}
