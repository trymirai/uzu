use serde::{Deserialize, Serialize};

use crate::units::Joules;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct EnergyMetrics {
    pub cpu: Joules,
    pub gpu: Joules,
    pub gpu_sram: Joules,
    pub ane: Joules,
    pub ram: Joules,

    pub package: Joules,
}

impl EnergyMetrics {
    pub fn total(&self) -> Joules {
        Joules(self.cpu.value() + self.gpu.value() + self.ane.value() + self.ram.value())
    }
}
