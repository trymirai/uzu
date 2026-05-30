use std::path::Path;

use crate::backends::common::{Backend, Context};

#[derive(Debug, Clone, Copy, Default)]
pub enum AsyncBatchSize {
    #[default]
    Default,
    Custom(usize),
}

impl AsyncBatchSize {
    pub fn resolve<B: Backend>(
        &self,
        model_path: &Path,
        context: &B::Context,
    ) -> Result<usize, B::Error> {
        match self {
            AsyncBatchSize::Default => context.recommended_async_batch_size(model_path),
            AsyncBatchSize::Custom(value) => Ok(*value),
        }
    }
}
