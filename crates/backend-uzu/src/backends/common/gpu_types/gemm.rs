use bitflags::bitflags;
use derive_more::Display;

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmBPrologueKind {
    FullPrecision,
    ScaleBiasDequant,
    ScaleZeroPointDequant,
    ScaleSymmetricDequant,
    LloydMaxDequant,
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GemmDTransform: u32 {
        const SCALE      = 1 << 0;
        const ACCUMULATE = 1 << 1;
        const BIAS       = 1 << 2;
        const RHT        = 1 << 3;
    }
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum GemmTiling {
    Tile8x32x32_Simdgroups1x1,
    Tile64x32x32_Simdgroups2x2,
    Tile64x64x16_Simdgroups2x2,
    Tile64x64x32_Simdgroups2x2,
    Tile32x32x32_Simdgroups2x2,
    Tile32x64x256_Simdgroups2x2,
    Tile64x32x256_Simdgroups4x1,
    Tile64x64x256_Simdgroups2x2,
    Tile128x128x256_Simdgroups4x4,
}

const MXU_SIMDGROUP_BLOCK_K: u32 = 32;

impl GemmTiling {
    pub const fn block_m(self) -> u32 {
        match self {
            Self::Tile8x32x32_Simdgroups1x1 => 8,
            Self::Tile64x32x32_Simdgroups2x2 => 64,
            Self::Tile64x64x16_Simdgroups2x2 => 64,
            Self::Tile64x64x32_Simdgroups2x2 => 64,
            Self::Tile32x32x32_Simdgroups2x2 => 32,
            Self::Tile32x64x256_Simdgroups2x2 => 32,
            Self::Tile64x32x256_Simdgroups4x1 => 64,
            Self::Tile64x64x256_Simdgroups2x2 => 64,
            Self::Tile128x128x256_Simdgroups4x4 => 128,
        }
    }

    pub const fn block_n(self) -> u32 {
        match self {
            Self::Tile8x32x32_Simdgroups1x1 => 32,
            Self::Tile64x32x32_Simdgroups2x2 => 32,
            Self::Tile64x64x16_Simdgroups2x2 => 64,
            Self::Tile64x64x32_Simdgroups2x2 => 64,
            Self::Tile32x32x32_Simdgroups2x2 => 32,
            Self::Tile32x64x256_Simdgroups2x2 => 64,
            Self::Tile64x32x256_Simdgroups4x1 => 32,
            Self::Tile64x64x256_Simdgroups2x2 => 64,
            Self::Tile128x128x256_Simdgroups4x4 => 128,
        }
    }

    pub const fn block_k(self) -> u32 {
        match self {
            Self::Tile8x32x32_Simdgroups1x1 => 32,
            Self::Tile64x32x32_Simdgroups2x2 => 32,
            Self::Tile64x64x16_Simdgroups2x2 => 16,
            Self::Tile64x64x32_Simdgroups2x2 => 32,
            Self::Tile32x32x32_Simdgroups2x2 => 32,
            Self::Tile32x64x256_Simdgroups2x2 => 256,
            Self::Tile64x32x256_Simdgroups4x1 => 256,
            Self::Tile64x64x256_Simdgroups2x2 => 256,
            Self::Tile128x128x256_Simdgroups4x4 => 256,
        }
    }

    pub const fn simdgroups_m(self) -> u32 {
        match self {
            Self::Tile8x32x32_Simdgroups1x1 => 1,
            Self::Tile64x32x32_Simdgroups2x2 => 2,
            Self::Tile64x64x16_Simdgroups2x2 => 2,
            Self::Tile64x64x32_Simdgroups2x2 => 2,
            Self::Tile32x32x32_Simdgroups2x2 => 2,
            Self::Tile32x64x256_Simdgroups2x2 => 2,
            Self::Tile64x32x256_Simdgroups4x1 => 4,
            Self::Tile64x64x256_Simdgroups2x2 => 2,
            Self::Tile128x128x256_Simdgroups4x4 => 4,
        }
    }

    pub const fn simdgroups_n(self) -> u32 {
        match self {
            Self::Tile8x32x32_Simdgroups1x1 => 1,
            Self::Tile64x32x32_Simdgroups2x2 => 2,
            Self::Tile64x64x16_Simdgroups2x2 => 2,
            Self::Tile64x64x32_Simdgroups2x2 => 2,
            Self::Tile32x32x32_Simdgroups2x2 => 2,
            Self::Tile32x64x256_Simdgroups2x2 => 2,
            Self::Tile64x32x256_Simdgroups4x1 => 1,
            Self::Tile64x64x256_Simdgroups2x2 => 2,
            Self::Tile128x128x256_Simdgroups4x4 => 4,
        }
    }

    pub const fn is_mxu_variant(self) -> bool {
        matches!(
            self,
            Self::Tile32x64x256_Simdgroups2x2
                | Self::Tile64x32x256_Simdgroups4x1
                | Self::Tile64x64x256_Simdgroups2x2
                | Self::Tile128x128x256_Simdgroups4x4
        )
    }

    pub const fn simdgroup_block_k(self) -> u32 {
        if self.is_mxu_variant() {
            MXU_SIMDGROUP_BLOCK_K
        } else {
            self.block_k()
        }
    }

    pub const fn fits_quant_group_size(
        self,
        group_size: u32,
    ) -> bool {
        match self {
            Self::Tile128x128x256_Simdgroups4x4 => group_size <= 64,
            _ => true,
        }
    }
}

pub const fn gemm_tiling_simdgroups_per_row(t: GemmTiling) -> u32 {
    t.simdgroups_m()
}
pub const fn gemm_tiling_simdgroups_per_column(t: GemmTiling) -> u32 {
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

impl GemmAlignment {
    pub fn new(
        m: bool,
        n: bool,
        k: bool,
    ) -> Self {
        let mut bits = Self::empty();
        bits.set(Self::M, m);
        bits.set(Self::N, n);
        bits.set(Self::K, k);
        bits
    }
}
