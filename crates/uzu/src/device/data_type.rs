use bytemuck::Pod;
use half::{bf16, f16};
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
