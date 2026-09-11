use safetensors::Dtype;

use super::Error;
use crate::data_type::DataType;

impl TryFrom<DataType> for Dtype {
    type Error = Error;

    fn try_from(data_type: DataType) -> Result<Self, Error> {
        Ok(match data_type {
            DataType::BF16 => Self::BF16,
            DataType::F16 => Self::F16,
            DataType::F32 => Self::F32,
            DataType::F64 => Self::F64,
            DataType::I8 => Self::I8,
            DataType::U8 => Self::U8,
            DataType::I16 => Self::I16,
            DataType::U16 => Self::U16,
            DataType::I32 => Self::I32,
            DataType::U32 => Self::U32,
            DataType::I64 => Self::I64,
            DataType::U64 => Self::U64,
            other @ (DataType::I4 | DataType::U4) => return Err(Error::UnsupportedDataType(other)),
        })
    }
}
