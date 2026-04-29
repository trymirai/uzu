use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ProgressCounters {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}
