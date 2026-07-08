use serde::{Deserialize, Serialize};

use crate::units::Rpm;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Fan {
    pub actual: Rpm,
    pub minimum: Rpm,
    pub maximum: Rpm,
    pub target: Rpm,
}
