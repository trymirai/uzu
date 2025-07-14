use serde::Deserialize;

use crate::session::config::common::ValueOrList;

#[derive(Clone, Deserialize, Debug)]
pub struct GenerationConfig {
    pub bos_token_id: Option<ValueOrList<u32>>,
    pub eos_token_id: Option<ValueOrList<u32>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
}
