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
            body: merge(context, record),
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

fn merge(
    context: &Value,
    record: &TelemetryRecord,
) -> Value {
    let mut body = context.clone();
    if let Value::Object(map) = &mut body {
        map.insert("event_time".to_string(), Value::String(record.event_time.clone()));
        if let Ok(Value::Object(event_map)) = serde_json::to_value(&record.event) {
            for (key, value) in event_map {
                map.insert(key, value);
            }
        }
    }
    body
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{super::record::TelemetryRecord, merge};
    use crate::telemetry::TelemetryEvent;

    #[test]
    fn merge_flattens_context_event_and_time() {
        let context = json!({ "device": { "os_name": "macOS" } });
        let record = TelemetryRecord {
            event_time: "2026-06-04T15:30:45.123Z".to_string(),
            event: TelemetryEvent::ModelDownloadFinished {
                model_id: "model-1".to_string(),
            },
        };
        let body = merge(&context, &record);
        assert_eq!(body["device"]["os_name"], json!("macOS"));
        assert_eq!(body["event_time"], json!("2026-06-04T15:30:45.123Z"));
        assert_eq!(body["type"], json!("model_download_finished"));
        assert_eq!(body["model_id"], json!("model-1"));
    }
}
