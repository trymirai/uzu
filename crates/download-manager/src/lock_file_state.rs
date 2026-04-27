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
