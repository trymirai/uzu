use serde::{Deserialize, Serialize};

use crate::units::{Megahertz, Percent};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    pub usage_percent: Percent, // combined, core-count weighted
    pub ecpu_frequency_megahertz: Megahertz,
    pub ecpu_usage_percent: Percent,
    pub pcpu_frequency_megahertz: Megahertz,
    pub pcpu_usage_percent: Percent,
}
