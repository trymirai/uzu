use derive_more::Display;

#[repr(C)]
#[allow(non_snake_case)]
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

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GemvTiling {
    Tg4x1_Sg1x32_Out4x4,
    Tg4x1_Sg1x32_Out1x4,
    Tg8x1_Sg1x32_Out4x4,
    Tg8x1_Sg1x32_Out1x4,
    Tg1x1_Sg8x4_Out4x4,
    Tg1x1_Sg8x4_Out1x4,
    Tg1x8_Sg1x32_Out4x4,
    Tg1x8_Sg1x32_Out1x4,
}

impl GemvTiling {
    pub const fn tg_simd_rows(self) -> u32 {
        match self {
            Self::Tg4x1_Sg1x32_Out4x4 | Self::Tg4x1_Sg1x32_Out1x4 => 4,
            Self::Tg8x1_Sg1x32_Out4x4 | Self::Tg8x1_Sg1x32_Out1x4 => 8,
            Self::Tg1x1_Sg8x4_Out4x4 | Self::Tg1x1_Sg8x4_Out1x4 => 1,
            Self::Tg1x8_Sg1x32_Out4x4 | Self::Tg1x8_Sg1x32_Out1x4 => 1,
        }
    }

    pub const fn tg_simd_cols(self) -> u32 {
        match self {
            Self::Tg1x8_Sg1x32_Out4x4 | Self::Tg1x8_Sg1x32_Out1x4 => 8,
            _ => 1,
        }
    }

    pub const fn sg_thread_rows(self) -> u32 {
        match self {
            Self::Tg1x1_Sg8x4_Out4x4 | Self::Tg1x1_Sg8x4_Out1x4 => 8,
            _ => 1,
        }
    }

    pub const fn sg_thread_cols(self) -> u32 {
        match self {
            Self::Tg1x1_Sg8x4_Out4x4 | Self::Tg1x1_Sg8x4_Out1x4 => 4,
            _ => 32,
        }
    }

    pub const fn thread_out_rows(self) -> u32 {
        match self {
            Self::Tg4x1_Sg1x32_Out1x4
            | Self::Tg8x1_Sg1x32_Out1x4
            | Self::Tg1x1_Sg8x4_Out1x4
            | Self::Tg1x8_Sg1x32_Out1x4 => 1,
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
