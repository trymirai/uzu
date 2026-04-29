use serde::{Deserialize, Serialize};

use crate::LockFileInfo;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LockFileState {
    Missing,
    OwnedByUs(LockFileInfo),
    OwnedBySameAppOldProcess(LockFileInfo),
    OwnedByOtherApp(LockFileInfo),
    Stale(LockFileInfo),
}

impl LockFileState {
    pub fn can_proceed(&self) -> bool {
        matches!(self, Self::Missing | Self::OwnedByUs(_) | Self::OwnedBySameAppOldProcess(_) | Self::Stale(_))
    }

    pub fn is_conflict(&self) -> bool {
        matches!(self, Self::OwnedByOtherApp(_))
    }
}
