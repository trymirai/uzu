use bon::Builder;
use nagare::api::Endpoint;
use serde::Serialize;

use super::{backend::Backend, types::Response};
use crate::device::Device;

#[derive(Serialize, Builder)]
pub struct FetchModels {
    device: CatalogDevice,
    backends: Vec<Backend>,
    #[builder(default)]
    include_traces: bool,
    #[builder(default)]
    show_all: bool,
}

#[derive(Clone, Serialize)]
pub struct CatalogDevice {
    os_name: Option<String>,
    cpu_name: Option<String>,
    memory_total: i64,
}

impl From<&Device> for CatalogDevice {
    fn from(device: &Device) -> Self {
        Self {
            os_name: device.os_name.clone(),
            cpu_name: device.cpu_name.clone(),
            memory_total: device.memory_total,
        }
    }
}

impl Endpoint for FetchModels {
    const PATH: &'static str = "fetch/models";

    type Request = Self;
    type Response = Response;
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn catalog_request_contains_hardware_metadata_but_no_local_paths() {
        let device = Device {
            os_name: Some("macOS".to_string()),
            cpu_name: Some("Apple silicon".to_string()),
            memory_total: 32_000_000_000,
            home_path: "/Users/private-user".to_string(),
        };
        let request = FetchModels::builder().device(CatalogDevice::from(&device)).backends(vec![]).build();

        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({
                "device": {
                    "os_name": "macOS",
                    "cpu_name": "Apple silicon",
                    "memory_total": 32_000_000_000i64,
                },
                "backends": [],
                "include_traces": false,
                "show_all": false,
            })
        );
    }
}
