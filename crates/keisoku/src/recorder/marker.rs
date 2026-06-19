use serde::{Deserialize, Serialize};

use crate::units::Milliseconds;

/// A timestamped annotation (e.g. an inference phase boundary).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker {
    pub elapsed_milliseconds: Milliseconds,
    pub label: String,
}
