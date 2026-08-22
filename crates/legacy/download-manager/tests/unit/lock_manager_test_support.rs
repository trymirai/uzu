use std::path::Path;

use uuid::Uuid;

use super::{DestinationLockLease, lock_path_for_destination};

impl DestinationLockLease {
    #[allow(dead_code)]
    pub(crate) async fn acquire_for_destination(
        destination_path: &Path,
        manager_id: &str,
        instance_id: Uuid,
    ) -> Result<Self, std::io::Error> {
        Self::acquire(&lock_path_for_destination(destination_path), manager_id, instance_id).await
    }
}
