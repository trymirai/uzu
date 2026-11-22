use bytemuck::{Pod, Zeroable};
use half::{bf16, f16};
use num_traits::{NumCast, ToPrimitive};
use uzu_engine_macros::uzu_config;

#[derive(Eq, PartialOrd, Ord, Copy, Hash)]
#[uzu_config]
pub enum DataType {
    BF16,
    F16,
    F32,
    F64,
    U2,
    I4,
    U4,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    /// float8 with 4 exponent and 3 mantissa bits, finite only (`float8_e4m3fn`), as safetensors names it.
    #[serde(rename = "F8_E4M3")]
    F8E4M3,
    /// float8 holding only a biased power-of-two exponent (`float8_e8m0fnu`), as safetensors names it.
    #[serde(rename = "F8_E8M0")]
    F8E8M0,
}

impl DataType {
    pub const fn size_in_bits(&self) -> usize {
        match self {
            DataType::U2 => 2,
            DataType::I4 | DataType::U4 => 4,
            DataType::I8 | DataType::U8 | DataType::F8E4M3 | DataType::F8E8M0 => 8,
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

pub trait DataTypeRepr: NumCast + Pod {
    fn data_type() -> DataType;
}

macro_rules! impl_data_type_repr {
    ($($type:ty => $variant:ident),+ $(,)?) => {
        $(
            impl DataTypeRepr for $type {
                fn data_type() -> DataType {
                    DataType::$variant
                }
            }
        )+
    };
}

/// Raw `float8_e4m3fn` value: 1 sign, 4 exponent (bias 7) and 3 mantissa bits, no infinities, NaN at 0x7F/0xFF.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct F8E4M3(pub u8);

impl F8E4M3 {
    pub fn decode(self) -> f32 {
        let sign = if self.0 & 0x80 != 0 {
            -1.0
        } else {
            1.0
        };
        let exponent = ((self.0 >> 3) & 0xF) as i32;
        let mantissa = (self.0 & 0x7) as f32;
        if exponent == 0xF && mantissa == 7.0 {
            f32::NAN
        } else if exponent == 0 {
            sign * mantissa / 8.0 * 2f32.powi(-6)
        } else {
            sign * (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
        }
    }
}

impl ToPrimitive for F8E4M3 {
    fn to_i64(&self) -> Option<i64> {
        self.to_f64()?.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.to_f64()?.to_u64()
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.decode() as f64)
    }
}

impl NumCast for F8E4M3 {
    /// Conversion into E4M3 needs rounding the engine never performs.
    fn from<T: ToPrimitive>(_: T) -> Option<Self> {
        None
    }
}

impl_data_type_repr! {
    F8E4M3 => F8E4M3,
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
