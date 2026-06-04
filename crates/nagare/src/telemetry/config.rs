use serde_json::Value;

use crate::api::Config;

const DEFAULT_CAPACITY: usize = 256;

pub struct TelemetryConfig {
    pub client: Config,
    pub path: String,
    pub context: Value,
    pub capacity: usize,
}

impl TelemetryConfig {
    pub fn new(
        client: Config,
        path: String,
        context: Value,
    ) -> Self {
        Self {
            client,
            path,
            context,
            capacity: DEFAULT_CAPACITY,
        }
    }
}
