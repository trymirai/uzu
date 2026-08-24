use bytemuck::Pod;
use half::{bf16, f16};
use num_traits::NumCast;

#[cfg(test)]
use crate::data_type::DataType;

// TODO: remove this

pub trait ArrayElement: NumCast + Pod {
    #[cfg(test)]
    fn data_type() -> DataType;
}

macro_rules! impl_array_element {
    ($($type:ty => $variant:ident),+ $(,)?) => {
        $(
            impl ArrayElement for $type {
                #[cfg(test)]
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
