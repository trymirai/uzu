use serde::{Deserialize, Serialize};

use crate::units::Watts;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PowerMetrics {
    pub cpu: Watts,
    pub gpu: Watts,
    pub gpu_sram: Watts,
    pub ane: Watts,
    pub ram: Watts,

    pub package: Watts,
}

impl PowerMetrics {
    pub fn total(&self) -> Watts {
        Watts(self.cpu.value() + self.gpu.value() + self.ane.value() + self.ram.value())
    }
}
