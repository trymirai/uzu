use serde::{Deserialize, Serialize};

use crate::LockFileInfo;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LockFileState {
    Missing,
    OwnedByUs(LockFileInfo),
    OwnedBySameAppOldProcess(LockFileInfo),
    OwnedByOtherApp(LockFileInfo),
    Stale(LockFileInfo),
    StaleUnparseable(Vec<u8>),
}

impl LockFileState {
    pub fn is_conflict(&self) -> bool {
        matches!(self, Self::OwnedByOtherApp(_))
    }
}
