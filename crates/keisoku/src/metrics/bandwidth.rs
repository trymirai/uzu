use serde::{Deserialize, Serialize};

use crate::units::GigabytesPerSecond;

/// DRAM memory bandwidth, from the IOReport `AMC Stats` DCS byte counters
/// (M1-M4) or the `PMP` `DRAM BW` residency histogram (M5+).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BandwidthMetrics {
    pub dram_read_gbps: GigabytesPerSecond,
    pub dram_write_gbps: GigabytesPerSecond,
}
