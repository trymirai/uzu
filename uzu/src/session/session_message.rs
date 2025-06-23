use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionMessageRole {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SessionMessage {
    pub role: SessionMessageRole,
    pub content: String,
}
