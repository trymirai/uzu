use serde::{Deserialize, Serialize};

use crate::LockFileState;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LockObservation {
    pub state: LockFileState,
}
