use serde::{Deserialize, Serialize};

use super::{Device, Marker};
use crate::snapshot::Snapshot;

/// A recorded telemetry session: device + the sampled time-series + markers.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Session {
    pub device: Device,
    pub interval_milliseconds: u64,
    pub snapshots: Vec<Snapshot>,
    pub markers: Vec<Marker>,
}

impl Session {
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn write_json(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<()> {
        let json = self.to_json().map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }
}
