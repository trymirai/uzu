use serde::{Deserialize, Serialize};

use crate::units::Milliseconds;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker {
    pub elapsed: Milliseconds,
    pub label: String,
}
