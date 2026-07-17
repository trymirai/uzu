use bitflags::bitflags;
use derive_more::Display;

/// Largest magnitude representable by symmetric int8 quantization.
pub const INT8_SYMMETRIC_QUANTIZATION_MAXIMUM: f32 = 127.0;

/// Inclusive asymmetric int8 bounds as magnitudes (signed min is `-MINIMUM_MAGNITUDE`).
pub const INT8_ASYMMETRIC_QUANTIZATION_MINIMUM_MAGNITUDE: f32 = 128.0;
pub const INT8_ASYMMETRIC_QUANTIZATION_MAXIMUM: f32 = 127.0;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ActivationPrepareOps: u32 {
        const INPUT_RHT = 1 << 0;
        const QUANTIZE = 1 << 1;
        const ASYMMETRIC = 1 << 2;
    }
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmAPrologueKind {
    FullPrecision,
    Int8Symmetric,
    Int8Asymmetric,
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationScaleStatistic {
    AbsMax,
    Rms,
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationQuantScheme {
    Symmetric,
    Asymmetric,
}
