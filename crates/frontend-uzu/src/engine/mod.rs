use message_processing::chat::ChatSession;
use uzu_types::{
    core::{InferenceBackend, InferenceModel},
    model::ModelSource,
};

use crate::engine::{config::EngineConfig, error::EngineError};

mod config;
mod error;

pub struct Engine {
    backends: Vec<Box<dyn InferenceBackend>>,
}

impl Engine {
    pub fn new() -> Result<Self, EngineError> {
        Self::new_with_config(EngineConfig::default())
    }

    pub fn new_with_config(config: EngineConfig) -> Result<Self, EngineError> {
        Ok(Self {
            backends: Vec::new(),
        })
    }

    pub fn add_backend(
        &mut self,
        backend: Box<dyn InferenceBackend>,
    ) -> Self {
        self.backends.push(backend);
        *self
    }
}

impl Engine {
    pub fn chat(model: ModelSource) -> Result<ChatSession, EngineError> {
        Ok(ChatSession::new())
    }
}
