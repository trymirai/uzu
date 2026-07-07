use serde::{Deserialize, Serialize};

use crate::units::{Megahertz, Percent};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    pub usage: Percent,
    pub ecpu_frequency: Megahertz,
    pub pcpu_frequency: Megahertz,
}
