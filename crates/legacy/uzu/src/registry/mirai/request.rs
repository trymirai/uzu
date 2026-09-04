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
