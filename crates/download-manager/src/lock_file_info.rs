use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LockFileInfo {
    pub manager_id: String,
    pub acquired_at: chrono::DateTime<chrono::Utc>,
    pub process_id: u32,
}

impl LockFileInfo {
    pub fn new(
        manager_id: String,
        process_id: u32,
    ) -> Self {
        Self {
            manager_id,
            acquired_at: chrono::Utc::now(),
            process_id,
        }
    }
}
