use serde_json::Value;

use crate::api::Endpoint;

/// `POST telemetry/events` — one event, already flattened into its context.
pub struct Events;

impl Endpoint for Events {
    const PATH: &'static str = "telemetry/events";

    /// The body is assembled by [`super::body::build_body`], which merges the
    /// event, its context and the timestamp into one flat object.
    type Request = Value;
    /// The server's reply is not read; [`crate::api::Client::send`] only checks
    /// the status.
    type Response = ();
}
