use serde::{Deserialize, Serialize};

use crate::units::{Megahertz, Percent};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub frequency: Megahertz,
    pub usage: Percent,
}
