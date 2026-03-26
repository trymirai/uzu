use serde::{Deserialize, Serialize};

use crate::DataType;

#[repr(C)]
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QuantizationMode {
    UINT4,
    INT8,
    UINT8,
}

impl QuantizationMode {
    pub fn packing_divisor(&self) -> usize {
        match self {
            QuantizationMode::UINT4 => 2,
            QuantizationMode::INT8 => 1,
            QuantizationMode::UINT8 => 1,
        }
    }

    pub fn storage_type(&self) -> DataType {
        match self {
            QuantizationMode::UINT4 => DataType::U8,
            QuantizationMode::INT8 => DataType::I8,
            QuantizationMode::UINT8 => DataType::U8,
        }
    }

    pub fn to_u32(&self) -> u32 {
        (*self).into()
    }
}

impl From<QuantizationMode> for DataType {
    fn from(val: QuantizationMode) -> Self {
        match val {
            QuantizationMode::UINT4 => DataType::U4,
            QuantizationMode::INT8 => DataType::I8,
            QuantizationMode::UINT8 => DataType::U8,
        }
    }
}

impl From<u32> for QuantizationMode {
    fn from(val: u32) -> Self {
        match val {
            0 => QuantizationMode::UINT4,
            1 => QuantizationMode::INT8,
            2 => QuantizationMode::UINT8,
            _ => panic!("Invalid QuantizationMode value: {val}"),
        }
    }
}

impl Into<u32> for QuantizationMode {
    fn into(self) -> u32 {
        match self {
            QuantizationMode::UINT4 => 0,
            QuantizationMode::INT8 => 1,
            QuantizationMode::UINT8 => 2,
        }
    }
}
