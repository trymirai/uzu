use std::path::Path;

use crate::{
    backends::common::{Backend, Context, DeviceClass},
    utils::ModelSize,
};

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
            AsyncBatchSize::Default => {
                let device_class = context.device_class();
                let model_size = ModelSize::from_path(model_path);
                default_batch_size(device_class, model_size)
            },
            AsyncBatchSize::Custom(value) => *value,
        }
    }
}

fn default_batch_size(
    device: DeviceClass,
    model_size: ModelSize,
) -> usize {
    match (model_size, device.is_high_end(), device) {
        (ModelSize::Large, true, _) => 32,
        (ModelSize::Large, _, DeviceClass::IPhone) => 4,
        (ModelSize::Large, _, _) => 8,
        (ModelSize::Small, true, _) => 256,
        (ModelSize::Small, _, DeviceClass::IPhone) => 8,
        (ModelSize::Small, _, _) => 64,
    }
}
