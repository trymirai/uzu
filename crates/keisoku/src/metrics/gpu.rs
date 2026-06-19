use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GpuMetrics {
    pub frequency_megahertz: u32,
    pub usage_percent: f32,
}
