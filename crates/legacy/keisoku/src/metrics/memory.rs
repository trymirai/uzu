use serde::{Deserialize, Serialize};

use crate::units::Bytes;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub ram_total: Bytes,
    pub ram_usage: Bytes,
    pub swap_total: Bytes,
    pub swap_usage: Bytes,
}
