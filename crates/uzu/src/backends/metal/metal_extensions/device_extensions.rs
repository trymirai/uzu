use byte_unit::Byte;
use metal::MTLDevice;
use objc2::{Message, msg_send, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceGeneration {
    Gen13,
    Gen14,
    Gen15,
    Gen16,
    Gen17,
    Gen18,
    Unknown(u8),
}

impl DeviceGeneration {
    pub fn generation_number(&self) -> u8 {
        match self {
            Self::Gen13 => 13,
            Self::Gen14 => 14,
            Self::Gen15 => 15,
            Self::Gen16 => 16,
            Self::Gen17 => 17,
            Self::Gen18 => 18,
            Self::Unknown(n) => *n,
        }
    }

    pub(crate) fn from_generation_number(generation_number: u8) -> Self {
        match generation_number {
            13 => Self::Gen13,
            14 => Self::Gen14,
            15 => Self::Gen15,
            16 => Self::Gen16,
            17 => Self::Gen17,
            18 => Self::Gen18,
            _ => Self::Unknown(generation_number),
        }
    }
}

pub trait DeviceExt: MTLDevice + Message {
    /// Human-readable chip name, e.g. "M2 Max", "M4 Pro", "A17 Pro".
    fn family_name(&self) -> String {
        let ns: Retained<NSString> = unsafe { msg_send![self, familyName] };
        ns.to_string()
    }

    /// Number of GPU shader cores.
    fn gpu_core_count(&self) -> u32 {
        unsafe { msg_send![self, gpuCoreCount] }
    }

    /// Total unified (shared) memory.
    fn shared_memory_size(&self) -> Byte {
        Byte::from_u64(unsafe { msg_send![self, sharedMemorySize] })
    }

    /// Whether the GPU supports SIMD group (warp-level) operations.
    fn supports_simd_group(&self) -> bool {
        unsafe { msg_send![self, supportsSIMDGroup] }
    }

    /// Whether the GPU supports `simdgroup_matrix` (8x8 matrix multiply).
    fn supports_simd_group_matrix(&self) -> bool {
        unsafe { msg_send![self, supportsSIMDGroupMatrix] }
    }

    /// Whether the GPU supports SIMD reduction operations.
    fn supports_simd_reduction(&self) -> bool {
        unsafe { msg_send![self, supportsSIMDReduction] }
    }

    /// Whether the GPU supports SIMD shuffle-and-fill operations.
    fn supports_simd_shuffle_and_fill(&self) -> bool {
        unsafe { msg_send![self, supportsSIMDShuffleAndFill] }
    }

    /// Whether the GPU supports SIMD shuffles and broadcast.
    fn supports_simd_shuffles_and_broadcast(&self) -> bool {
        unsafe { msg_send![self, supportsSIMDShufflesAndBroadcast] }
    }

    /// Whether the GPU has a Matrix eXtension Unit (neural accelerator for compute).
    /// True on M5+ (Gen18+), false on M1-M4.
    fn supports_mxu(&self) -> bool {
        unsafe { msg_send![self, supportsMXU] }
    }

    /// Whether the GPU supports Thread-Local Storage.
    fn supports_tls(&self) -> bool {
        unsafe { msg_send![self, supportsTLS] }
    }
}

impl DeviceExt for ProtocolObject<dyn MTLDevice> {}
