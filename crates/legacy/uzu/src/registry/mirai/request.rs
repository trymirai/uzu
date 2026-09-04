use bon::Builder;
use serde::{Deserialize, Serialize};

use crate::device::Device;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Backend {
    pub identifier: String,
    pub version: String,
}

#[derive(Serialize, Builder)]
pub struct FetchModelsRequest {
    device: Device,
    backends: Vec<Backend>,
    #[builder(default)]
    include_traces: bool,
    #[builder(default)]
    show_all: bool,
}

#[cfg(test)]
mod tests {
    use serde_json::{json, to_value};

    use super::{Backend, FetchModelsRequest};
    use crate::device::Device;

    #[test]
    fn fetch_models_request_serializes_the_api_payload() {
        let request = FetchModelsRequest::builder()
            .device(Device {
                os_name: Some("macOS".to_string()),
                cpu_name: Some("Apple".to_string()),
                memory_total: 32,
                home_path: "/tmp".to_string(),
            })
            .backends(vec![Backend {
                identifier: "uzu".to_string(),
                version: "1".to_string(),
            }])
            .show_all(true)
            .build();

        assert_eq!(
            to_value(request).expect("request should serialize"),
            json!({
                "device": {
                    "os_name": "macOS",
                    "cpu_name": "Apple",
                    "memory_total": 32,
                    "home_path": "/tmp"
                },
                "backends": [{ "identifier": "uzu", "version": "1" }],
                "include_traces": false,
                "show_all": true
            })
        );
    }
}
