use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LockFileInfo {
    pub manager_id: String,
    #[serde(default)]
    pub instance_id: Uuid,
    pub acquired_at: chrono::DateTime<chrono::Utc>,
    pub process_id: u32,
}

impl LockFileInfo {
    pub fn new(
        manager_id: String,
        instance_id: Uuid,
        process_id: u32,
    ) -> Self {
        Self {
            manager_id,
            instance_id,
            acquired_at: chrono::Utc::now(),
            process_id,
        }
    }
}
