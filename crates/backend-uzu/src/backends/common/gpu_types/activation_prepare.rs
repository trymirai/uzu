use bitflags::bitflags;

pub const INT8_SYMMETRIC_QUANTIZATION_MAXIMUM: f32 = 127.0;
pub const ACTIVATION_QUANTIZATION_GROUP_SIZE: u32 = 32;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ActivationPrepareOps: u32 {
        const INPUT_RHT = 1 << 0;
        const QUANTIZE = 1 << 1;
    }
}
