use bitflags::bitflags;
use derive_more::Display;

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmInputPrologueKind {
    FullPrecision,
    ExternalRht,
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmOutputTransformKind {
    Store,
    Scale,
    Accumulate,
    Bias,
    Rht,
    ScaleAccumulate,
    ScaleAccumulateBias,
    ScaleAccumulateBiasRht,
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmWeightPrologueKind {
    FullPrecision,
    ScaleBiasDequant,
    ScaleZeroPointDequant,
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmTiling {
    T8x32x32_1x1,
    T64x32x32_2x2,
    T64x64x16_2x2,
    T64x64x32_2x2,
    T32x32x32_2x2,
    T32x64x32_2x2,
    T64x32x32_4x1,
    T128x128x32_4x4,
}

impl GemmTiling {
    pub const fn block_m(self) -> u32 {
        match self {
            Self::T8x32x32_1x1 => 8,
            Self::T64x32x32_2x2 => 64,
            Self::T64x64x16_2x2 => 64,
            Self::T64x64x32_2x2 => 64,
            Self::T32x32x32_2x2 => 32,
            Self::T32x64x32_2x2 => 32,
            Self::T64x32x32_4x1 => 64,
            Self::T128x128x32_4x4 => 128,
        }
    }

    pub const fn block_n(self) -> u32 {
        match self {
            Self::T8x32x32_1x1 => 32,
            Self::T64x32x32_2x2 => 32,
            Self::T64x64x16_2x2 => 64,
            Self::T64x64x32_2x2 => 64,
            Self::T32x32x32_2x2 => 32,
            Self::T32x64x32_2x2 => 64,
            Self::T64x32x32_4x1 => 32,
            Self::T128x128x32_4x4 => 128,
        }
    }

    pub const fn block_k(self) -> u32 {
        match self {
            Self::T8x32x32_1x1 => 32,
            Self::T64x32x32_2x2 => 32,
            Self::T64x64x16_2x2 => 16,
            Self::T64x64x32_2x2 => 32,
            Self::T32x32x32_2x2 => 32,
            Self::T32x64x32_2x2 => 32,
            Self::T64x32x32_4x1 => 32,
            Self::T128x128x32_4x4 => 32,
        }
    }

    pub const fn simdgroups_m(self) -> u32 {
        match self {
            Self::T8x32x32_1x1 => 1,
            Self::T64x32x32_2x2 => 2,
            Self::T64x64x16_2x2 => 2,
            Self::T64x64x32_2x2 => 2,
            Self::T32x32x32_2x2 => 2,
            Self::T32x64x32_2x2 => 2,
            Self::T64x32x32_4x1 => 4,
            Self::T128x128x32_4x4 => 4,
        }
    }

    pub const fn simdgroups_n(self) -> u32 {
        match self {
            Self::T8x32x32_1x1 => 1,
            Self::T64x32x32_2x2 => 2,
            Self::T64x64x16_2x2 => 2,
            Self::T64x64x32_2x2 => 2,
            Self::T32x32x32_2x2 => 2,
            Self::T32x64x32_2x2 => 2,
            Self::T64x32x32_4x1 => 1,
            Self::T128x128x32_4x4 => 4,
        }
    }
}

pub const fn gemm_tiling_smg_m(t: GemmTiling) -> u32 {
    t.simdgroups_m()
}
pub const fn gemm_tiling_smg_n(t: GemmTiling) -> u32 {
    t.simdgroups_n()
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GemmAlignment: u32 {
        const M = 1 << 0;
        const N = 1 << 1;
        const K = 1 << 2;
    }
}
