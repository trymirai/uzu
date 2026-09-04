use nagare::api::Endpoint;

use super::{request::FetchModelsRequest, types::Response};

/// `POST fetch/models` — the model catalog for this device and backend set.
pub struct FetchModels;

impl Endpoint for FetchModels {
    const PATH: &'static str = "fetch/models";

    type Request = FetchModelsRequest;
    type Response = Response;
}
