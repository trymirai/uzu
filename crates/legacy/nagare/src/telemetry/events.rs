use serde_json::Value;

use crate::api::Endpoint;

pub struct Events;

impl Endpoint for Events {
    const PATH: &'static str = "telemetry/events";

    type Request = Value;
    type Response = ();
}
