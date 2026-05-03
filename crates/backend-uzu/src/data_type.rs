use bytemuck::Pod;
use half::{bf16, f16};
use num_traits::NumCast;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
pub enum DataType {
    // Floating point
    #[serde(rename = "bfloat16")]
    BF16,
    #[serde(rename = "float16")]
    F16,
    #[serde(rename = "float32")]
    F32,
    #[serde(rename = "float64")]
    F64,
    // Sub-byte integers
    #[serde(rename = "int4")]
    I4,
    #[serde(rename = "uint4")]
    U4,
    // Normal integers
    #[serde(rename = "int8")]
    I8,
    #[serde(rename = "uint8")]
    U8,
    #[serde(rename = "int16")]
    I16,
    #[serde(rename = "uint16")]
    U16,
    #[serde(rename = "int32")]
    I32,
    #[serde(rename = "uint32")]
    U32,
    #[serde(rename = "int64")]
    I64,
    #[serde(rename = "uint64")]
    U64,
}

impl DataType {
    pub const fn size_in_bits(&self) -> usize {
        match self {
            DataType::I4 | DataType::U4 => 4,
            DataType::I8 | DataType::U8 => 8,
            DataType::I16 | DataType::U16 => 16,
            DataType::BF16 | DataType::F16 => 16,
            DataType::F32 | DataType::I32 | DataType::U32 => 32,
            DataType::F64 | DataType::I64 | DataType::U64 => 64,
        }
    }

    pub const fn size_in_bytes(&self) -> usize {
        self.size_in_bits().div_ceil(8)
    }
}

pub trait ArrayElement: NumCast + Pod {
    fn data_type() -> DataType;
}

macro_rules! impl_array_element {
    ($($type:ty => $variant:ident),+ $(,)?) => {
        $(
            impl ArrayElement for $type {
                fn data_type() -> DataType {
                    DataType::$variant
                }
            }
        )+
    };
}

impl_array_element! {
    f16 => F16,
    bf16 => BF16,
    f32 => F32,
    f64 => F64,
    i8 => I8,
    u8 => U8,
    i16 => I16,
    u16 => U16,
    i32 => I32,
    u32 => U32,
    i64 => I64,
    u64 => U64,
}
