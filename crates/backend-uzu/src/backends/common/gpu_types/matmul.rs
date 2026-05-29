#![allow(non_snake_case)]

use derive_more::Display;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct GemmParams {
    pub M: u32,
    pub N: u32,
    pub K: u32,
    pub leading_dimension_a: u32,
    pub leading_dimension_b: u32,
    pub leading_dimension_d: u32,
    pub threadgroups_per_column: u32,
    pub threadgroups_per_row: u32,
    pub aligned_inner_iterations: u32,
    pub use_morton: bool,
    pub ab_scale: f32,
}

#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
pub struct GemvParams {
    pub in_vec_size: u32,
    pub out_vec_size: u32,
    pub batch_size: u32,
    pub matrix_leading_dimension: u32,
    pub output_rows_per_threadgroup: u32,
    pub ab_scale: f32,
}

/// Per-threadgroup tile layout for the GEMV kernel. `*Narrow` siblings drop
/// `thread_out_rows` to 1 for tiny output dimensions; `Wide` doubles as the
/// canonical quantized tile.
///
/// | variant      | tg rows×cols | sg rows×cols | out rows×cols |
/// |--------------|--------------|--------------|---------------|
/// | Standard     | 4×1          | 1×32         | 4×4           |
/// | Wide         | 8×1          | 1×32         | 4×4           |
/// | SmallInput   | 1×1          | 8×4          | 4×4           |
/// | SplitInput   | 1×8          | 1×32         | 4×4           |
#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemvTiling {
    Standard,
    StandardNarrow,
    Wide,
    WideNarrow,
    SmallInput,
    SmallInputNarrow,
    SplitInput,
    SplitInputNarrow,
}

impl GemvTiling {
    pub const fn tg_simd_rows(self) -> u32 {
        match self {
            Self::Standard | Self::StandardNarrow => 4,
            Self::Wide | Self::WideNarrow => 8,
            Self::SmallInput | Self::SmallInputNarrow => 1,
            Self::SplitInput | Self::SplitInputNarrow => 1,
        }
    }

    pub const fn tg_simd_cols(self) -> u32 {
        match self {
            Self::SplitInput | Self::SplitInputNarrow => 8,
            _ => 1,
        }
    }

    pub const fn sg_thread_rows(self) -> u32 {
        match self {
            Self::SmallInput | Self::SmallInputNarrow => 8,
            _ => 1,
        }
    }

    pub const fn sg_thread_cols(self) -> u32 {
        match self {
            Self::SmallInput | Self::SmallInputNarrow => 4,
            _ => 32,
        }
    }

    pub const fn thread_out_rows(self) -> u32 {
        match self {
            Self::StandardNarrow | Self::WideNarrow | Self::SmallInputNarrow | Self::SplitInputNarrow => 1,
            _ => 4,
        }
    }

    pub const fn thread_out_cols(self) -> u32 {
        4
    }

    pub const fn output_rows_per_threadgroup(self) -> u32 {
        self.tg_simd_rows() * self.sg_thread_rows() * self.thread_out_rows()
    }
}
