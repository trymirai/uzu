use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EncodingName {
    #[serde(rename = "gpt-oss")]
    GptOss,
}
