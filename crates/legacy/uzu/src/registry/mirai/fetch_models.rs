use bon::Builder;
use nagare::api::Endpoint;
use serde::Serialize;

use super::{backend::Backend, types::Response};
use crate::device::Device;

#[derive(Serialize, Builder)]
pub struct FetchModels {
    device: Device,
    backends: Vec<Backend>,
    #[builder(default)]
    include_traces: bool,
    #[builder(default)]
    show_all: bool,
}

impl Endpoint for FetchModels {
    const PATH: &'static str = "fetch/models";

    type Request = Self;
    type Response = Response;
}
