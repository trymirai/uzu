use serde_json::Value;
use tokio::sync::mpsc;
#[cfg(not(target_family = "wasm"))]
use tokio::sync::mpsc::channel as TokioMpscChannel;

use super::{TelemetryEvent, record::TelemetryRecord};
#[cfg(not(target_family = "wasm"))]
use crate::api::Client;
use crate::api::Config;

#[cfg(not(target_family = "wasm"))]
const CAPACITY: usize = 256;

#[derive(Clone)]
pub struct Telemetry {
    sender: Option<mpsc::Sender<TelemetryRecord>>,
}

impl Telemetry {
    pub fn disabled() -> Self {
        Self {
            sender: None,
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn new(
        client_config: Config,
        path: String,
        context: Value,
    ) -> Self {
        let client = match Client::new(client_config) {
            Ok(client) => client,
            Err(error) => {
                tracing::warn!(%error, "telemetry disabled: failed to build client");
                return Self::disabled();
            },
        };
        let (sender, receiver) = TokioMpscChannel::<TelemetryRecord>(CAPACITY);
        tokio::spawn(super::worker::run(client, path, context, receiver));
        Self {
            sender: Some(sender),
        }
    }

    #[cfg(target_family = "wasm")]
    pub fn new(
        _client_config: Config,
        _path: String,
        _context: Value,
    ) -> Self {
        Self::disabled()
    }

    pub fn report(
        &self,
        event: TelemetryEvent,
    ) {
        if let Some(sender) = &self.sender {
            let _ = sender.try_send(TelemetryRecord::new(event));
        }
    }
}
