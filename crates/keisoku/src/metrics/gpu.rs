use serde::{Deserialize, Serialize};

use crate::units::{Megahertz, Percent};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub frequency_megahertz: Megahertz,
    pub usage_percent: Percent,
}
