use serde::{Deserialize, Serialize};

use crate::units::{GigabytesPerSecond, Percent, Watts};

/// Apple Neural Engine activity. `active` comes from PMP state residency (the
/// only reliable signal on M5+, where the Energy Model ANE power channel reads
/// zero); on earlier chips it falls back to `power / 8 W` (mactop's
/// `aneMaxPowerW`). Bandwidth is the AMC `ANE RD/WR` byte counters (M1-M4) or
/// the PMP `AF BW` residency histogram (M5+).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct NeuralEngineMetrics {
    pub power: Watts,
    pub active: Percent,
    pub read_bandwidth: GigabytesPerSecond,
    pub write_bandwidth: GigabytesPerSecond,
}
