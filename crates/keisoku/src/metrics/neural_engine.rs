use serde::{Deserialize, Serialize};

/// Apple Neural Engine activity. `active_percent` comes from PMP state residency
/// (the only reliable signal on M5+, where the Energy Model ANE power channel
/// reads zero); on earlier chips it falls back to `power / 8 W` (mactop's
/// `aneMaxPowerW`). Bandwidth is the AMC `ANE RD/WR` byte counters (M1-M4) or
/// the PMP `AF BW` residency histogram (M5+).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct NeuralEngineMetrics {
    pub power_watts: f32,
    pub active_percent: f32,
    pub read_bandwidth_gbps: f32,
    pub write_bandwidth_gbps: f32,
}
