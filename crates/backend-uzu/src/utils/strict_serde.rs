use std::io::Read;

use serde::{
    Deserialize, Serialize,
    de::{DeserializeOwned, Error as _},
};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum Unsupported {}

pub fn from_reader_strict<T: DeserializeOwned>(reader: impl Read) -> Result<T, serde_json::Error> {
    let value: serde_json::Value = serde_json::from_reader(reader)?;
    let mut unknown: Vec<String> = Vec::new();
    let parsed = serde_ignored::deserialize(value, |path| unknown.push(path.to_string()))?;
    if unknown.is_empty() {
        Ok(parsed)
    } else {
        Err(serde_json::Error::custom(format!("unknown config fields: {}", unknown.join(", "))))
    }
}
