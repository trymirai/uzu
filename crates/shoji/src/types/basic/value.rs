use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[bindings::export(Struct, name = "Value")]
#[derive(Clone, PartialEq)]
pub struct Value {
    pub json: String,
}

impl fmt::Debug for Value {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        let serde_value: serde_json::Value = serde_json::from_str(&self.json).map_err(|_| fmt::Error)?;
        let pretty_json = serde_json::to_string_pretty(&serde_value).map_err(|_| fmt::Error)?;
        write!(formatter, "{pretty_json}")
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let serde_value: serde_json::Value = serde_json::from_str(&self.json).map_err(serde::ser::Error::custom)?;
        serde_value.serialize(serializer)
    }
}

impl<'d> Deserialize<'d> for Value {
    fn deserialize<D: Deserializer<'d>>(deserializer: D) -> Result<Self, D::Error> {
        let serde_value = serde_json::Value::deserialize(deserializer)?;
        Ok(Value {
            json: serde_value.to_string(),
        })
    }
}

impl From<serde_json::Value> for Value {
    fn from(value: serde_json::Value) -> Self {
        Self {
            json: value.to_string(),
        }
    }
}

impl TryFrom<Value> for serde_json::Value {
    type Error = serde_json::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        serde_json::from_str(&value.json)
    }
}
