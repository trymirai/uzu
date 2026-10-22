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
        const BIAS       = 1 << 1;
        const RHT        = 1 << 2;
        const SOFT_CAP   = 1 << 3;
    }
}
