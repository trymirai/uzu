use serde::{Deserialize, Serialize};

use crate::units::GigabytesPerSecond;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BandwidthMetrics {
    pub dram_read: GigabytesPerSecond,
    pub dram_write: GigabytesPerSecond,
    pub ane_read: GigabytesPerSecond,
    pub ane_write: GigabytesPerSecond,
}
