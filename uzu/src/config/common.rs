use serde::{Deserialize, Serialize};

use crate::DataType;

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename = "DataType")]
#[serde(rename_all = "lowercase")]
pub enum ConfigDataType {
    BFloat16,
    Float16,
    Float32,
    Int4,
    Int8,
}

impl Into<DataType> for ConfigDataType {
    fn into(self) -> DataType {
        match self {
            ConfigDataType::BFloat16 => DataType::BF16,
            ConfigDataType::Float16 => DataType::F16,
            ConfigDataType::Float32 => DataType::F32,
            ConfigDataType::Int4 => DataType::I4,
            ConfigDataType::Int8 => DataType::I8,
        }
    }
}

impl From<DataType> for ConfigDataType {
    fn from(dtype: DataType) -> Self {
        match dtype {
            DataType::BF16 => ConfigDataType::BFloat16,
            DataType::F16 => ConfigDataType::Float16,
            DataType::F32 => ConfigDataType::Float32,
            DataType::I4 => ConfigDataType::Int4,
            DataType::I8 => ConfigDataType::Int8,
            _ => panic!("Unsupported data type: {0:?}", dtype),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationMode {
    UInt4,
    Int8,
    UInt8,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Copy, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    SILU,
    GELU,
}
