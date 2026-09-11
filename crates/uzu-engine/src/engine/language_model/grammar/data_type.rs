use xgrammar::{DLDataType, DLDataTypeCode};

use crate::data_type::DataType;

trait DLDataTypeCodeProvider {
    fn dl_data_type_code(self) -> DLDataTypeCode;
}

impl DLDataTypeCodeProvider for DataType {
    fn dl_data_type_code(self) -> DLDataTypeCode {
        match self {
            DataType::BF16 => DLDataTypeCode::kDLBfloat,
            DataType::F16 | DataType::F32 | DataType::F64 => DLDataTypeCode::kDLFloat,
            DataType::I4 | DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => DLDataTypeCode::kDLInt,
            DataType::U4 | DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64 => DLDataTypeCode::kDLUInt,
        }
    }
}

impl From<DataType> for DLDataType {
    fn from(data_type: DataType) -> Self {
        Self {
            code: data_type.dl_data_type_code() as u8,
            bits: data_type.size_in_bits() as u8,
            lanes: 1,
        }
    }
}
