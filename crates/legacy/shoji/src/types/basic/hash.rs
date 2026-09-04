use serde::{Deserialize, Serialize};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HashMethod {
    #[serde(rename = "crc32c")]
    CRC32C,
    #[serde(rename = "md5")]
    MD5,
}

#[bindings::export(Structure(Class))]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Hash {
    pub method: HashMethod,
    pub value: String,
}
