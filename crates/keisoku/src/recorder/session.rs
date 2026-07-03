use crate::{snapshot::Snapshot, units::Milliseconds};

/// Snapshots collected between `start` and `stop`, with the sampling interval.
#[derive(Debug, Default, Clone)]
pub struct Session {
    pub interval: Milliseconds,
    pub snapshots: Vec<Snapshot>,
}
