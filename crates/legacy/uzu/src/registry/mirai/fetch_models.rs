use nagare::api::Endpoint;

use super::{request::FetchModelsRequest, types::Response};

pub struct FetchModels;

impl Endpoint for FetchModels {
    const PATH: &'static str = "fetch/models";

    type Request = FetchModelsRequest;
    type Response = Response;
}
