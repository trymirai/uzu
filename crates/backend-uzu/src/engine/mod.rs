use std::sync::Arc;

use thiserror::Error;

use crate::{
    backends::common::{Backend, Context},
    engine::capture::CaptureManager,
};

pub mod classifier_model;
pub mod language_model;

mod capture;

pub struct Engine<B: Backend> {
    context: Arc<B::Context>,
    capture_manager: Option<CaptureManager<B>>,
}

#[derive(Debug, Error)]
pub enum EngineNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
}

impl<B: Backend> Engine<B> {
    pub fn new() -> Result<Arc<Self>, EngineNewError<B>> {
        let capture_enabled = CaptureManager::<B>::pre_load_enable();

        let context = <B::Context as Context>::new().map_err(EngineNewError::Backend)?;

        let capture_manager = capture_enabled.then(|| CaptureManager::new(context.clone()));

        Ok(Arc::new(Self {
            context,
            capture_manager,
        }))
    }

    pub fn peak_memory_usage(&self) -> Option<usize> {
        self.context.peak_memory_usage()
    }
}
