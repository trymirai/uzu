use serde::{Deserialize, Serialize};

/// A timestamped annotation (e.g. an inference phase boundary).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker {
    pub elapsed_milliseconds: u64,
    pub label: String,
}
