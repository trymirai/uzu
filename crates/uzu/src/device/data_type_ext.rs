use xgrammar::{DLDataType, DLDataTypeCode};

use crate::DataType;

// Standard conversion from crate DataType to DLDataType
impl Into<DLDataType> for DataType {
    fn into(self) -> DLDataType {
        match self {
            DataType::BF16 => DLDataType {
                code: DLDataTypeCode::kDLBfloat as u8,
                bits: 16,
                lanes: 1,
            },
            DataType::F16 => DLDataType {
                code: DLDataTypeCode::kDLFloat as u8,
                bits: 16,
                lanes: 1,
            },
            DataType::F32 => DLDataType {
                code: DLDataTypeCode::kDLFloat as u8,
                bits: 32,
                lanes: 1,
            },
            DataType::F64 => DLDataType {
                code: DLDataTypeCode::kDLFloat as u8,
                bits: 64,
                lanes: 1,
            },
            DataType::I4 => DLDataType {
                code: DLDataTypeCode::kDLInt as u8,
                bits: 4,
                lanes: 1,
            },
            DataType::U4 => DLDataType {
                code: DLDataTypeCode::kDLUInt as u8,
                bits: 4,
                lanes: 1,
            },
            DataType::I8 => DLDataType {
                code: DLDataTypeCode::kDLInt as u8,
                bits: 8,
                lanes: 1,
            },
            DataType::U8 => DLDataType {
                code: DLDataTypeCode::kDLUInt as u8,
                bits: 8,
                lanes: 1,
            },
            DataType::I16 => DLDataType {
                code: DLDataTypeCode::kDLInt as u8,
                bits: 16,
                lanes: 1,
            },
            DataType::U16 => DLDataType {
                code: DLDataTypeCode::kDLUInt as u8,
                bits: 16,
                lanes: 1,
            },
            DataType::I32 => DLDataType {
                code: DLDataTypeCode::kDLInt as u8,
                bits: 32,
                lanes: 1,
            },
            DataType::U32 => DLDataType {
                code: DLDataTypeCode::kDLUInt as u8,
                bits: 32,
                lanes: 1,
            },
            DataType::I64 => DLDataType {
                code: DLDataTypeCode::kDLInt as u8,
                bits: 64,
                lanes: 1,
            },
            DataType::U64 => DLDataType {
                code: DLDataTypeCode::kDLUInt as u8,
                bits: 64,
                lanes: 1,
            },
        }
    }
}
