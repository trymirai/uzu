use std::collections::HashMap;
use std::ops::Index;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Double(f64),
    String(String),
    Array(Vec<Value>),
    Object(HashMap<String, Value>),
}

impl Value {
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    pub fn is_object(&self) -> bool {
        matches!(self, Value::Object(_))
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Value::Array(_))
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Value::String(_))
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(value) => Some(*value),
            Value::UInt(value) if *value <= i64::MAX as u64 => {
                Some(*value as i64)
            },
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::UInt(value) => Some(*value),
            Value::Int(value) if *value >= 0 => Some(*value as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Double(value) => Some(*value),
            Value::Int(value) => Some(*value as f64),
            Value::UInt(value) => Some(*value as f64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&HashMap<String, Value>> {
        match self {
            Value::Object(value) => Some(value),
            _ => None,
        }
    }

    pub fn get_by_key(
        &self,
        key: &str,
    ) -> Option<&Value> {
        match self {
            Value::Object(value) => value.get(key),
            _ => None,
        }
    }

    pub fn get_by_index(
        &self,
        index: usize,
    ) -> Option<&Value> {
        match self {
            Value::Array(value) => value.get(index),
            _ => None,
        }
    }
}

impl Index<&str> for Value {
    type Output = Value;

    fn index(
        &self,
        key: &str,
    ) -> &Self::Output {
        self.get_by_key(key).unwrap_or(&Value::Null)
    }
}

impl Index<usize> for Value {
    type Output = Value;

    fn index(
        &self,
        index: usize,
    ) -> &Self::Output {
        self.get_by_index(index).unwrap_or(&Value::Null)
    }
}

impl PartialEq<&str> for Value {
    fn eq(
        &self,
        other: &&str,
    ) -> bool {
        match self {
            Value::String(value) => value == *other,
            _ => false,
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::Null
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::Bool(value)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Int(value)
    }
}

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Value::Int(value as i64)
    }
}

impl From<u64> for Value {
    fn from(value: u64) -> Self {
        Value::UInt(value)
    }
}

impl From<u32> for Value {
    fn from(value: u32) -> Self {
        Value::UInt(value as u64)
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Double(value)
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::String(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Value::String(value.to_string())
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(value: Vec<T>) -> Self {
        Value::Array(value.into_iter().map(Into::into).collect())
    }
}

impl<T: Into<Value>> From<HashMap<String, T>> for Value {
    fn from(map: HashMap<String, T>) -> Self {
        Value::Object(
            map.into_iter().map(|(key, value)| (key, value.into())).collect(),
        )
    }
}

impl From<serde_json::Value> for Value {
    fn from(serde_value: serde_json::Value) -> Self {
        match serde_value {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(value) => Value::Bool(value),
            serde_json::Value::Number(value) => {
                if let Some(number) = value.as_i64() {
                    Value::Int(number)
                } else if let Some(number) = value.as_u64() {
                    Value::UInt(number)
                } else if let Some(number) = value.as_f64() {
                    Value::Double(number)
                } else {
                    Value::Null
                }
            },
            serde_json::Value::String(value) => Value::String(value),
            serde_json::Value::Array(value) => {
                Value::Array(value.into_iter().map(Value::from).collect())
            },
            serde_json::Value::Object(map) => Value::Object(
                map.into_iter()
                    .map(|(key, value)| (key, Value::from(value)))
                    .collect(),
            ),
        }
    }
}

impl From<Value> for serde_json::Value {
    fn from(tool_value: Value) -> Self {
        match tool_value {
            Value::Null => serde_json::Value::Null,
            Value::Bool(value) => serde_json::Value::Bool(value),
            Value::Int(value) => serde_json::Value::Number(value.into()),
            Value::UInt(value) => serde_json::Value::Number(value.into()),
            Value::Double(value) => serde_json::Number::from_f64(value)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Value::String(value) => serde_json::Value::String(value),
            Value::Array(value) => serde_json::Value::Array(
                value.into_iter().map(serde_json::Value::from).collect(),
            ),
            Value::Object(object) => serde_json::Value::Object(
                object
                    .into_iter()
                    .map(|(key, value)| (key, serde_json::Value::from(value)))
                    .collect(),
            ),
        }
    }
}
