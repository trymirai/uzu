use std::rc::Rc;

use thiserror::Error;

use crate::backends::common::{Backend, Context};

pub mod classifier_model;
pub mod dflash_speculator;
pub mod language_model;

pub struct Engine<B: Backend> {
    context: Rc<B::Context>,
}

#[derive(Debug, Error)]
pub enum EngineNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
}

impl<B: Backend> Engine<B> {
    pub fn new() -> Result<Self, EngineNewError<B>> {
        let context = <B::Context as Context>::new().map_err(EngineNewError::Backend)?;

        Ok(Self {
            context,
        })
    }

    pub fn peak_memory_usage(&self) -> Option<usize> {
        self.context.peak_memory_usage()
    }
}
