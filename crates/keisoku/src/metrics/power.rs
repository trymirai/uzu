use serde::{Deserialize, Serialize};

use crate::units::Watts;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PowerMetrics {
    pub cpu: Watts,
    pub gpu: Watts,
    pub gpu_sram: Watts,
    pub ane: Watts,
    pub ram: Watts,
    /// Sum of the component rails above.
    pub total: Watts,
    /// Whole-package power from the SMC `PSTR` rail; falls back to `total`.
    pub package: Watts,
}
