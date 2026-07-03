use serde::{Deserialize, Serialize};

use crate::units::{Megahertz, Percent};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    pub usage: Percent,
    pub ecpu_frequency: Megahertz,
    pub ecpu_usage: Percent,
    pub pcpu_frequency: Megahertz,
    pub pcpu_usage: Percent,
    pub per_core: Box<[Percent]>,
}
