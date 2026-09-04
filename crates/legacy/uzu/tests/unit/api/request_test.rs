use serde_json::{json, to_value};

use super::*;
use crate::{device::Device, registry::mirai::Backend};

#[test]
fn fetch_models_request_serializes_the_api_payload() {
    let request = FetchModelsRequest::new(
        Device {
            os_name: Some("macOS".to_string()),
            cpu_name: Some("Apple".to_string()),
            memory_total: 32,
            home_path: "/tmp".to_string(),
        },
        vec![Backend {
            identifier: "uzu".to_string(),
            version: "1".to_string(),
        }],
        false,
        true,
    );

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
