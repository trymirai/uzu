#[cfg(not(target_family = "wasm"))]
use std::time::Duration;

#[cfg(not(target_family = "wasm"))]
use bon::bon;
use tokio::sync::mpsc;
#[cfg(not(target_family = "wasm"))]
use tokio::sync::mpsc::channel as TokioMpscChannel;

#[cfg(not(target_family = "wasm"))]
use super::TelemetryContext;
use super::{TelemetryEvent, record::TelemetryRecord};
#[cfg(not(target_family = "wasm"))]
use crate::api::{Client, RetryConfig};

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

    pub fn report(
        &self,
        event: TelemetryEvent,
    ) {
        if let Some(sender) = &self.sender {
            let _ = sender.try_send(TelemetryRecord::new(event));
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[bon]
impl Telemetry {
    #[builder]
    pub fn new(
        #[builder(into)] base_url: String,
        context: TelemetryContext,
    ) -> Self {
        // Telemetry is a background best-effort sender, so it waits far longer
        // than a startup-path caller would tolerate.
        let retry = RetryConfig {
            max_attempts: 5,
            base_delay: Duration::from_secs(1),
            budget: Duration::from_secs(60),
        };
        let client = match Client::builder().base_url(base_url).retry(retry).build() {
            Ok(client) => client,
            Err(error) => {
                tracing::warn!(%error, "telemetry disabled: failed to build client");
                return Self::disabled();
            },
        };
        let (sender, receiver) = TokioMpscChannel::<TelemetryRecord>(CAPACITY);
        tokio::spawn(super::worker::run(client, context, receiver));
        Self {
            sender: Some(sender),
        }
    }
}
