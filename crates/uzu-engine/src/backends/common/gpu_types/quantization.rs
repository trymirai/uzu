use derive_more::Display;
use uzu_engine_macros::uzu_config;

use crate::data_type::DataType;

#[repr(C)]
#[derive(Display, Copy, Eq, Hash)]
#[uzu_config]
pub enum QuantizationMode {
    #[serde(rename = "uint4")]
    U4,
    #[serde(rename = "int8")]
    I8,
    #[serde(rename = "uint8")]
    U8,
    #[serde(rename = "uint2")]
    U2,
}

impl QuantizationMode {
    pub fn from_unsigned_bits(bits: u32) -> Option<Self> {
        match bits {
            2 => Some(Self::U2),
            4 => Some(Self::U4),
            8 => Some(Self::U8),
            _ => None,
        }
    }

    pub fn packing_divisor(&self) -> u32 {
        let bits = DataType::from(*self).size_in_bits() as u32;
        assert_eq!(8 % bits, 0, "QuantizationMode bit width ({bits}) must divide 8 evenly");
        8 / bits
    }

    pub fn storage_type(&self) -> DataType {
        match self {
            QuantizationMode::U2 | QuantizationMode::U4 => DataType::U8,
            QuantizationMode::I8 => DataType::I8,
            QuantizationMode::U8 => DataType::U8,
        }
    }

    pub fn weight_codes_sign_flip_mask(&self) -> Option<u8> {
        match self {
            QuantizationMode::U2 => None,
            QuantizationMode::U4 => Some(0x88),
            QuantizationMode::I8 => None,
            QuantizationMode::U8 => Some(0x80),
        }
    }
}

impl From<QuantizationMode> for DataType {
    fn from(val: QuantizationMode) -> Self {
        match val {
            QuantizationMode::U2 => DataType::U2,
            QuantizationMode::U4 => DataType::U4,
            QuantizationMode::I8 => DataType::I8,
            QuantizationMode::U8 => DataType::U8,
        }
    }
}
