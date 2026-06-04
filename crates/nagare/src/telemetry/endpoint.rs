use indexmap::IndexMap;
use reqwest::{Method, header::CONTENT_TYPE};
use serde_json::Value;

use super::record::TelemetryRecord;
use crate::api::{Config, Endpoint, Payload};

pub(super) struct TelemetryEndpoint {
    path: String,
    body: Value,
}

impl TelemetryEndpoint {
    pub(super) fn new(
        path: String,
        context: &Value,
        record: &TelemetryRecord,
    ) -> Self {
        Self {
            path,
            body: build_body(context, record),
        }
    }
}

impl Endpoint for TelemetryEndpoint {
    fn method(&self) -> Method {
        Method::POST
    }

    fn path(&self) -> String {
        self.path.clone()
    }

    fn headers(&self) -> IndexMap<String, String> {
        IndexMap::from([(CONTENT_TYPE.to_string(), "application/json".to_string())])
    }

    fn payload(
        &self,
        _: &Config,
    ) -> Payload {
        Payload {
            query: None,
            body: Some(self.body.clone()),
        }
    }
}

fn build_body(
    context: &Value,
    record: &TelemetryRecord,
) -> Value {
    let mut body = serde_json::to_value(&record.event).unwrap_or_else(|_| Value::Object(Default::default()));
    if let Value::Object(map) = &mut body {
        map.insert("event_time".to_string(), Value::String(record.event_time.clone()));
        map.insert("device".to_string(), context.clone());
    }
    body
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{super::record::TelemetryRecord, build_body};
    use crate::telemetry::TelemetryEvent;

    #[test]
    fn build_body_nests_event_time_device_and_payload() {
        let device = json!({ "os_name": "macOS" });
        let record = TelemetryRecord {
            event_time: "2026-06-04T16:24:29Z".to_string(),
            event: TelemetryEvent::ModelInferenceFinished {
                model_id: "model-1".to_string(),
                stats: Default::default(),
            },
        };
        let body = build_body(&device, &record);
        assert_eq!(body["event_name"], json!("model_inference_finished"));
        assert_eq!(body["event_time"], json!("2026-06-04T16:24:29Z"));
        assert_eq!(body["device"]["os_name"], json!("macOS"));
        assert_eq!(body["payload"]["model_id"], json!("model-1"));
        assert!(body["payload"]["stats"].is_object());
    }

    #[test]
    fn build_body_wraps_failure_error() {
        let record = TelemetryRecord {
            event_time: "2026-06-04T16:24:29Z".to_string(),
            event: TelemetryEvent::ModelInferenceFailed {
                error: json!({ "message": "boom" }),
            },
        };
        let body = build_body(&json!({}), &record);
        assert_eq!(body["event_name"], json!("model_inference_failed"));
        assert_eq!(body["payload"]["error"]["message"], json!("boom"));
    }
}
