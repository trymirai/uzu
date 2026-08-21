use serde::{Deserialize, Serialize};
use shoji::types::basic::TokenId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokensConfig {
    pub bos_token_id: Option<TokenId>,
    pub eos_token_id: Option<TokenId>,
    pub stop_token_ids: Vec<TokenId>,
    pub banned_token_ids: Vec<TokenId>,
}
