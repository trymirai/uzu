use crate::snapshot::Snapshot;

/// Snapshots collected between `start` and `stop`.
#[derive(Debug, Default, Clone)]
pub struct Session {
    pub snapshots: Vec<Snapshot>,
}
