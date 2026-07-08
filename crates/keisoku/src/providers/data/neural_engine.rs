use serde::{Deserialize, Serialize};

use crate::units::Percent;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct NeuralEngineMetrics {
    pub active: Percent,
}
