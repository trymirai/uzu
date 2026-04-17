use std::path::Path;

use crate::backends::common::{Backend, Context};

#[derive(Debug, Clone, Copy)]
pub enum AsyncBatchSize {
    Default,
    Custom(usize),
}

impl Default for AsyncBatchSize {
    fn default() -> Self {
        AsyncBatchSize::Default
    }
}

impl AsyncBatchSize {
    pub fn resolve<B: Backend>(
        &self,
        model_path: &Path,
        context: &B::Context,
    ) -> usize {
        match self {
            AsyncBatchSize::Default => context.recommended_async_batch_size(model_path),
            AsyncBatchSize::Custom(value) => *value,
        }
    }
}
