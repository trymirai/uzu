use serde::{Deserialize, Serialize};

use crate::units::Rpm;

/// One fan's speeds, read from the SMC `F{i}Ac/Mn/Mx/Tg` keys.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Fan {
    pub actual: Rpm,
    pub minimum: Rpm,
    pub maximum: Rpm,
    pub target: Rpm,
}

/// All fans reported by the SMC (empty on fanless machines / off macOS).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FanMetrics {
    pub fans: Vec<Fan>,
}
