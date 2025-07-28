use bytemuck::Pod;
use half::{bf16, f16};
use mpsgraph::DataType as MPSDataType;
use num_traits::NumCast;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
pub enum DataType {
    // Floating point
    BF16,
    F16,
    F32,
    F64,
    // Sub-byte integers
    I4,
    U4,
    // Normal integers
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
}

impl DataType {
    pub fn size_in_bits(&self) -> usize {
        match self {
            DataType::BF16 => 16,
            DataType::F16 => 16,
            DataType::F32 => 32,
            DataType::F64 => 64,
            DataType::I4 => 4,
            DataType::U4 => 4,
            DataType::I8 => 8,
            DataType::U8 => 8,
            DataType::I16 => 16,
            DataType::U16 => 16,
            DataType::I32 => 32,
            DataType::U32 => 32,
            DataType::I64 => 64,
            DataType::U64 => 64,
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        (self.size_in_bits() + 7) / 8 // Round up to nearest byte
    }
}

impl From<MPSDataType> for DataType {
    fn from(dtype: MPSDataType) -> Self {
        match dtype {
            MPSDataType::BFloat16 => DataType::BF16,
            MPSDataType::Float16 => DataType::F16,
            MPSDataType::Float32 => DataType::F32,
            MPSDataType::Int4 => DataType::I4,
            MPSDataType::UInt4 => DataType::U4,
            MPSDataType::Int8 => DataType::I8,
            MPSDataType::UInt8 => DataType::U8,
            MPSDataType::Int16 => DataType::I16,
            MPSDataType::UInt16 => DataType::U16,
            MPSDataType::Int32 => DataType::I32,
            MPSDataType::UInt32 => DataType::U32,
            MPSDataType::Int64 => DataType::I64,
            MPSDataType::UInt64 => DataType::U64,
            _ => panic!("Unsupported MPS data type: {0:?}", dtype),
        }
    }
}

impl From<DataType> for MPSDataType {
    fn from(dtype: DataType) -> Self {
        match dtype {
            DataType::BF16 => MPSDataType::BFloat16,
            DataType::F16 => MPSDataType::Float16,
            DataType::F32 => MPSDataType::Float32,
            DataType::I4 => MPSDataType::Int4,
            DataType::U4 => MPSDataType::UInt4,
            DataType::I8 => MPSDataType::Int8,
            DataType::U8 => MPSDataType::UInt8,
            DataType::I16 => MPSDataType::Int16,
            DataType::U16 => MPSDataType::UInt16,
            DataType::I32 => MPSDataType::Int32,
            DataType::U32 => MPSDataType::UInt32,
            DataType::I64 => MPSDataType::Int64,
            DataType::U64 => MPSDataType::UInt64,
            _ => panic!("Unsupported MPS data type: {0:?}", dtype),
        }
    }
}

pub trait ArrayElement: NumCast + Pod {
    fn data_type() -> DataType;
}

impl ArrayElement for f16 {
    fn data_type() -> DataType {
        DataType::F16
    }
}

impl ArrayElement for bf16 {
    fn data_type() -> DataType {
        DataType::BF16
    }
}

impl ArrayElement for f32 {
    fn data_type() -> DataType {
        DataType::F32
    }
}

impl ArrayElement for f64 {
    fn data_type() -> DataType {
        DataType::F64
    }
}

impl ArrayElement for i8 {
    fn data_type() -> DataType {
        DataType::I8
    }
}

impl ArrayElement for u8 {
    fn data_type() -> DataType {
        DataType::U8
    }
}

impl ArrayElement for i16 {
    fn data_type() -> DataType {
        DataType::I16
    }
}

impl ArrayElement for u16 {
    fn data_type() -> DataType {
        DataType::U16
    }
}

impl ArrayElement for i32 {
    fn data_type() -> DataType {
        DataType::I32
    }
}

impl ArrayElement for u32 {
    fn data_type() -> DataType {
        DataType::U32
    }
}

impl ArrayElement for i64 {
    fn data_type() -> DataType {
        DataType::I64
    }
}

impl ArrayElement for u64 {
    fn data_type() -> DataType {
        DataType::U64
    }
}
