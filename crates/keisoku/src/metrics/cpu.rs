use serde::{Deserialize, Serialize};

use crate::units::{Megahertz, Percent};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    pub usage: Percent, // combined, core-count weighted
    pub ecpu_frequency: Megahertz,
    pub ecpu_usage: Percent,
    pub pcpu_frequency: Megahertz,
    pub pcpu_usage: Percent,
    pub per_core: Vec<Percent>, // one entry per logical core (macOS)
}
