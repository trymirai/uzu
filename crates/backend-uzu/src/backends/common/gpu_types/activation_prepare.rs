use bitflags::bitflags;
use derive_more::Display;

/// Largest magnitude representable by symmetric int8 quantization.
pub const INT8_SYMMETRIC_QUANTIZATION_MAXIMUM: f32 = 127.0;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ActivationPrepareOps: u32 {
        const INPUT_RHT = 1 << 0;
        const QUANTIZE = 1 << 1;
    }
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmAPrologueKind {
    FullPrecision,
    Int8Symmetric,
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationScaleStatistic {
    AbsMax,
    Rms,
}
