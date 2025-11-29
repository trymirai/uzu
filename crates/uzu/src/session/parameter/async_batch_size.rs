use crate::session::parameter::ResolvableValue;

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

impl ResolvableValue<usize> for AsyncBatchSize {
    fn resolve(&self) -> usize {
        match self {
            AsyncBatchSize::Default => 128,
            AsyncBatchSize::Custom(value) => *value,
        }
    }
}
