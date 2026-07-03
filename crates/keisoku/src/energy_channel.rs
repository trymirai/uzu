use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyModelChannel {
    pub name: String,
    pub unit: String,
    pub value: i64,
}
