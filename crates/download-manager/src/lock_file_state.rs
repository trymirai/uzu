use crate::LockFileInfo;

#[derive(Debug, Clone, PartialEq)]
pub enum LockFileState {
    /// No lock file exists - safe to acquire
    Missing,

    /// Lock owned by this manager instance (same manager_id + same PID)
    OwnedByUs(LockFileInfo),

    /// Lock owned by same app but old process (same manager_id + different PID, process dead)
    OwnedBySameAppOldProcess(LockFileInfo),

    /// Lock owned by another active app (different manager_id, active)
    OwnedByOtherApp(LockFileInfo),

    /// Lock exists but is stale (different manager_id, timeout expired)
    Stale(LockFileInfo),
}

impl LockFileState {
    /// Check if we can safely acquire or use this lock
    pub fn can_proceed(&self) -> bool {
        matches!(
            self,
            LockFileState::Missing
                | LockFileState::OwnedByUs(_)
                | LockFileState::OwnedBySameAppOldProcess(_)
                | LockFileState::Stale(_)
        )
    }

    /// Check if this is a blocking conflict
    pub fn is_conflict(&self) -> bool {
        matches!(self, LockFileState::OwnedByOtherApp(_))
    }
}
