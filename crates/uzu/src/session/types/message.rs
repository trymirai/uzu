use serde::{Deserialize, Serialize};

use crate::session::types::Role;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}
