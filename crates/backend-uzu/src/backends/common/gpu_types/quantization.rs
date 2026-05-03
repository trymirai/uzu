use serde::{Deserialize, Serialize};

use crate::DataType;

#[repr(C)]
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum QuantizationMode {
    #[serde(rename = "uint4")]
    U4,
    #[serde(rename = "int8")]
    I8,
    #[serde(rename = "uint8")]
    U8,
}

impl QuantizationMode {
    pub fn packing_divisor(&self) -> usize {
        match self {
            QuantizationMode::U4 => 2,
            QuantizationMode::I8 => 1,
            QuantizationMode::U8 => 1,
        }
    }

    pub fn storage_type(&self) -> DataType {
        match self {
            QuantizationMode::U4 => DataType::U8,
            QuantizationMode::I8 => DataType::I8,
            QuantizationMode::U8 => DataType::U8,
        }
    }

    pub fn to_u32(&self) -> u32 {
        (*self).into()
    }
}

impl From<QuantizationMode> for DataType {
    fn from(val: QuantizationMode) -> Self {
        match val {
            QuantizationMode::U4 => DataType::U4,
            QuantizationMode::I8 => DataType::I8,
            QuantizationMode::U8 => DataType::U8,
        }
    }
}

impl From<u32> for QuantizationMode {
    fn from(val: u32) -> Self {
        match val {
            0 => QuantizationMode::U4,
            1 => QuantizationMode::I8,
            2 => QuantizationMode::U8,
            _ => panic!("Invalid QuantizationMode value: {val}"),
        }
    }
}

impl Into<u32> for QuantizationMode {
    fn into(self) -> u32 {
        match self {
            QuantizationMode::U4 => 0,
            QuantizationMode::I8 => 1,
            QuantizationMode::U8 => 2,
        }
    }
}
