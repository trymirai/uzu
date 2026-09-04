use indexmap::IndexMap;
use nagare::api::{Config, Endpoint, Payload};
use reqwest::{Method, header::CONTENT_TYPE};
use serde::Serialize;
use serde_json::json;

use crate::{device::Device, registry::mirai::Backend};

#[derive(Serialize)]
pub struct FetchModelsRequest {
    device: Device,
    backends: Vec<Backend>,
    include_traces: bool,
    show_all: bool,
}

impl FetchModelsRequest {
    pub fn new(
        device: Device,
        backends: Vec<Backend>,
        include_traces: bool,
        show_all: bool,
    ) -> Self {
        Self {
            device,
            backends,
            include_traces,
            show_all,
        }
    }
}

impl Endpoint for FetchModelsRequest {
    fn method(&self) -> Method {
        Method::POST
    }

    fn path(&self) -> String {
        "fetch/models".to_string()
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
            body: Some(json!(self)),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/api/request_test.rs"]
mod tests;
