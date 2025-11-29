use std::path::Path;

use crate::utils::{DeviceClass, ModelSize};

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
    pub fn resolve(
        &self,
        model_path: &Path,
    ) -> usize {
        match self {
            AsyncBatchSize::Default => {
                let device = DeviceClass::detect();
                let model_size = ModelSize::from_path(model_path);
                default_batch_size(device, model_size)
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
