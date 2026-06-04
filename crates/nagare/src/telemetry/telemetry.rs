use tokio::sync::mpsc;

use super::{TelemetryConfig, TelemetryEvent, record::TelemetryRecord};
#[cfg(not(target_family = "wasm"))]
use crate::api::Client;

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
    pub fn new(config: TelemetryConfig) -> Self {
        let client = match Client::new(config.client) {
            Ok(client) => client,
            Err(error) => {
                tracing::warn!(%error, "telemetry disabled: failed to build client");
                return Self::disabled();
            },
        };
        let (sender, receiver) = mpsc::channel::<TelemetryRecord>(config.capacity);
        tokio::spawn(super::worker::run(client, config.path, config.context, receiver));
        Self {
            sender: Some(sender),
        }
    }

    #[cfg(target_family = "wasm")]
    pub fn new(_config: TelemetryConfig) -> Self {
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
