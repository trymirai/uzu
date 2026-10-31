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

#[derive(Clone, Default, Serialize)]
pub struct CatalogDevice {
    os_name: Option<String>,
    cpu_name: Option<String>,
    memory_total: Option<i64>,
}

impl CatalogDevice {
    pub(super) fn new(
        device: &Device,
        share_device_metadata: bool,
    ) -> Self {
        if share_device_metadata {
            Self {
                os_name: device.os_name.clone(),
                cpu_name: device.cpu_name.clone(),
                memory_total: device.memory_total,
            }
        } else {
            Self::default()
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
    fn device_metadata_is_omitted_by_default() {
        let device = Device {
            os_name: Some("macOS".to_string()),
            cpu_name: Some("Apple silicon".to_string()),
            memory_total: Some(32_000_000_000),
            home_path: "/Users/private-user".to_string(),
        };
        let request = FetchModels::builder().device(CatalogDevice::new(&device, false)).backends(vec![]).build();

        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({
                "device": {
                    "os_name": null,
                    "cpu_name": null,
                    "memory_total": null,
                },
                "backends": [],
                "include_traces": false,
                "show_all": false,
            })
        );
    }

    #[test]
    fn device_metadata_can_be_shared_explicitly_without_local_paths() {
        let device = Device {
            os_name: Some("macOS".to_string()),
            cpu_name: Some("Apple silicon".to_string()),
            memory_total: Some(32_000_000_000),
            home_path: "/Users/private-user".to_string(),
        };
        let request = FetchModels::builder().device(CatalogDevice::new(&device, true)).backends(vec![]).build();

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
