use serde::{Deserialize, Serialize};

use crate::units::Watts;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PowerMetrics {
    pub cpu_watts: Watts,
    pub gpu_watts: Watts,
    pub gpu_sram_watts: Watts,
    pub ane_watts: Watts,
    pub ram_watts: Watts,
    pub total_watts: Watts,
}
