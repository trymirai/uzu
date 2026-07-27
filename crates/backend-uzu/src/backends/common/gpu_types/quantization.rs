use derive_more::Display;
use proc_macros::uzu_config;

use crate::data_type::DataType;

#[repr(C)]
#[derive(Display, Copy, Eq, Hash)]
#[uzu_config]
pub enum QuantizationMode {
    #[serde(rename = "uint4")]
    U4,
    /// Offset-binary U8 codes stored as `bitcast_i8(code ^ 0x80)`; scales and
    /// affine metadata retain their U8-code semantics.
    #[serde(rename = "int8")]
    I8,
    #[serde(rename = "uint8")]
    U8,
}

impl QuantizationMode {
    #[must_use]
    pub const fn from_storage(
        bits: u32,
        storage_type: DataType,
    ) -> Option<Self> {
        match (bits, storage_type) {
            (4, DataType::U8) => Some(Self::U4),
            (8, DataType::I8) => Some(Self::I8),
            (8, DataType::U8) => Some(Self::U8),
            _ => None,
        }
    }

    pub fn packing_divisor(&self) -> usize {
        let bits = DataType::from(*self).size_in_bits();
        assert_eq!(8 % bits, 0, "QuantizationMode bit width ({bits}) must divide 8 evenly");
        8 / bits
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
