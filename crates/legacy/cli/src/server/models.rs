use rocket::{State, get, serde::json::Json};
use serde::{Deserialize, Serialize};

use crate::server::{ServerState, request_info::RequestInfo};

#[derive(Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<Model>,
}

#[get("/models")]
#[tracing::instrument(skip_all, parent = &request_info.span)]
pub fn handle_models(
    state: &State<ServerState>,
    request_info: &RequestInfo,
) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![Model {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "uzu".to_string(),
        }],
    })
}
