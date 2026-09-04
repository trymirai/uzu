use tokio::sync::mpsc;

use super::{TelemetryContext, body::build_body, events::Events, record::TelemetryRecord};
use crate::api::Client;

pub(super) async fn run(
    client: Client,
    context: TelemetryContext,
    mut receiver: mpsc::Receiver<TelemetryRecord>,
) {
    while let Some(record) = receiver.recv().await {
        // The client retries transient failures itself; anything surfacing here
        // is either fatal or out of budget, so the event is dropped.
        if let Err(error) = client.send::<Events>(&build_body(&context, &record)).await {
            tracing::warn!(%error, "telemetry event dropped");
        }
    }
}
