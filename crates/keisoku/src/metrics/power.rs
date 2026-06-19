use serde::{Deserialize, Serialize};

use crate::units::Watts;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PowerMetrics {
    pub cpu: Watts,
    pub gpu: Watts,
    pub gpu_sram: Watts,
    pub ane: Watts,
    pub ram: Watts,

    pub total: Watts,

    pub package: Watts,
}
