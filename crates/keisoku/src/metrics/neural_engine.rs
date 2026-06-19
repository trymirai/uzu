use serde::{Deserialize, Serialize};

use crate::units::{GigabytesPerSecond, Percent, Watts};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct NeuralEngineMetrics {
    pub power: Watts,
    pub active: Percent,
    pub read_bandwidth: GigabytesPerSecond,
    pub write_bandwidth: GigabytesPerSecond,
}
