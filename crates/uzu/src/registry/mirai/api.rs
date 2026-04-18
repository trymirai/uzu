use indexmap::IndexMap;
use reqwest::{Method, header::CONTENT_TYPE};
use serde_json::json;
use shoji::api::{Config, Endpoint as EndpointTrait, Payload};

use crate::{device::Device, registry::mirai::config::Backend};

pub enum Endpoint {
    FetchModels {
        device: Device,
        backends: Vec<Backend>,
        include_traces: bool,
    },
}

impl EndpointTrait for Endpoint {
    fn method(&self) -> Method {
        match self {
            Endpoint::FetchModels {
                ..
            } => Method::POST,
        }
    }

    fn path(&self) -> String {
        match self {
            Endpoint::FetchModels {
                ..
            } => "fetch/models".to_string(),
        }
    }

    fn headers(&self) -> IndexMap<String, String> {
        IndexMap::from([(CONTENT_TYPE.to_string(), "application/json".to_string())])
    }

    fn payload(
        &self,
        _: &Config,
    ) -> Payload {
        match self {
            Endpoint::FetchModels {
                device,
                backends,
                include_traces,
            } => Payload {
                query: None,
                body: Some(json!({
                    "device": device,
                    "backends": backends,
                    "include_traces": include_traces,
                })),
            },
        }
    }
}
