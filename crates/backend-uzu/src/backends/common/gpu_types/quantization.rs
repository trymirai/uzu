use debug_display::Display;
use serde::{Deserialize, Serialize};

use crate::DataType;

#[repr(C)]
#[derive(Debug, Display, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
        8 / DataType::from(*self).size_in_bits()
    }

    pub fn storage_type(&self) -> DataType {
        match self {
            QuantizationMode::U4 => DataType::U8,
            QuantizationMode::I8 => DataType::I8,
            QuantizationMode::U8 => DataType::U8,
        }
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
