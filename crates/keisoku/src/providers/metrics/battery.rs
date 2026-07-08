use serde::{Deserialize, Serialize};

use crate::units::Percent;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BatteryMetrics {
    pub present: bool,
    pub percent: Percent,
    pub charging: bool,
    pub on_ac_power: bool,
}
