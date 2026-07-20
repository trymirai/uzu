use bitflags::bitflags;
use derive_more::Display;

pub const INT8_SYMMETRIC_QUANTIZATION_MAXIMUM: f32 = 127.0;
pub const ACTIVATION_QUANTIZATION_GROUP_SIZE: u32 = 32;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ActivationPrepareOps: u32 {
        const INPUT_RHT = 1 << 0;
        const QUANTIZE = 1 << 1;
        const ROW_SUMS = 1 << 3;
    }
}

#[repr(C)]
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GemmAPrologueKind {
    FullPrecision,
    Int8Symmetric,
}
