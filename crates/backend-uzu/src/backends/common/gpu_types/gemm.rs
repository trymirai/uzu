use bitflags::bitflags;
use derive_more::Display;

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmBPrologueKind {
    FullPrecision,
    ScaleBiasDequant,
    ScaleZeroPointDequant,
    ScaleSymmetricDequant,
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct GemmDTransform: u32 {
        const SCALE      = 1 << 0;
        const ACCUMULATE = 1 << 1;
        const BIAS       = 1 << 2;
        const RHT        = 1 << 3;
        const SOFT_CAP   = 1 << 4;
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
    Tile16x32x256_Simdgroups1x1,
    Tile16x128x256_Simdgroups1x4,
    Tile32x64x256_Simdgroups2x2,
    Tile64x32x256_Simdgroups4x1,
    Tile64x64x256_Simdgroups2x2,
    Tile128x128x256_Simdgroups4x4,
}

// Every tile's block and simdgroup dimensions, and whether it runs on the matrix units,
// are generated from its variant name -- see igata's `tile_geometry` module, which emits
// the same answers for Metal.
include!(concat!(env!("OUT_DIR"), "/tile_geometry.rs"));

impl GemmTiling {
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

/// Bit width of a quantized B operand. There is deliberately no zero variant: an
/// unquantized B is [`WeightsKey::FullPrecision`], which carries no width at all.
#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantBits {
    B4 = 4,
    B8 = 8,
}

/// Quantization group size. As with [`QuantBits`], zero is not a value.
#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantGroupSize {
    G16 = 16,
    G32 = 32,
    G64 = 64,
    G128 = 128,
}

/// The dequantizing B prologues: [`GemmBPrologueKind`] minus `FullPrecision`. Variant
/// names match that enum's, which is how the build script maps them onto the shader's
/// `B_PROLOGUE` axis.
#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantPrologue {
    ScaleBiasDequant = 1,
    ScaleZeroPointDequant = 2,
    ScaleSymmetricDequant = 3,
}

/// How a GEMM/GEMV reads its B operand.
///
/// The shader spells this as three independent template axes, which lets a caller
/// describe weights that are quantized to zero bits, or full precision with a group
/// size. Those combinations exist in neither the kernels nor reality; keeping the
/// concept a sum type is what stops them being expressible on either side.
#[proc_macros::variant_group(B_PROLOGUE, BITS, GROUP_SIZE)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightsKey {
    FullPrecision,
    Quant {
        b_prologue: QuantPrologue,
        bits: QuantBits,
        group_size: QuantGroupSize,
    },
}

impl QuantBits {
    pub fn new(bits: u32) -> Option<Self> {
        match bits {
            4 => Some(Self::B4),
            8 => Some(Self::B8),
            _ => None,
        }
    }
}

impl QuantGroupSize {
    pub fn new(group_size: u32) -> Option<Self> {
        match group_size {
            16 => Some(Self::G16),
            32 => Some(Self::G32),
            64 => Some(Self::G64),
            128 => Some(Self::G128),
            _ => None,
        }
    }
}

impl WeightsKey {
    pub fn b_prologue(self) -> GemmBPrologueKind {
        match self {
            Self::FullPrecision => GemmBPrologueKind::FullPrecision,
            Self::Quant {
                b_prologue: QuantPrologue::ScaleBiasDequant,
                ..
            } => GemmBPrologueKind::ScaleBiasDequant,
            Self::Quant {
                b_prologue: QuantPrologue::ScaleZeroPointDequant,
                ..
            } => GemmBPrologueKind::ScaleZeroPointDequant,
            Self::Quant {
                b_prologue: QuantPrologue::ScaleSymmetricDequant,
                ..
            } => GemmBPrologueKind::ScaleSymmetricDequant,
        }
    }

    pub fn bits(self) -> Option<u32> {
        match self {
            Self::FullPrecision => None,
            Self::Quant {
                bits,
                ..
            } => Some(bits as u32),
        }
    }

    pub fn group_size(self) -> Option<u32> {
        match self {
            Self::FullPrecision => None,
            Self::Quant {
                group_size,
                ..
            } => Some(group_size as u32),
        }
    }

    pub fn is_quantized(self) -> bool {
        !matches!(self, Self::FullPrecision)
    }
}
