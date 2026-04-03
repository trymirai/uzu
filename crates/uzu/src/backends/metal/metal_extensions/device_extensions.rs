use bytesize::ByteSize;
use metal::MTLDevice;
use objc2::{
    Message, msg_send,
    rc::Retained,
    runtime::{AnyObject, ProtocolObject},
    sel,
};
use objc2_foundation::NSString;

/// Check whether the concrete class of `obj` has `sel` in its method table.
fn class_responds_to<T: Message + ?Sized>(
    obj: &T,
    sel: objc2::runtime::Sel,
) -> bool {
    let obj_ptr: *const T = obj;
    let any: &AnyObject = unsafe { &*(obj_ptr as *const AnyObject) };
    let cls = any.class();
    let result = cls.responds_to(sel);
    eprintln!("[DeviceExt] class={:?} selector={:?} class_responds_to={}", cls.name(), sel.name(), result);
    result
}

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

pub trait DeviceExt: MTLDevice + Message + Sized {
    /// Human-readable chip name, e.g. "M2 Max", "M4 Pro", "A17 Pro".
    fn family_name(&self) -> String {
        if class_responds_to(self, sel!(familyName)) {
            let ns: Retained<NSString> = unsafe { msg_send![self, familyName] };
            ns.to_string()
        } else {
            "Unknown".into()
        }
    }

    /// Number of GPU shader cores.
    fn gpu_core_count(&self) -> u32 {
        if class_responds_to(self, sel!(gpuCoreCount)) {
            unsafe { msg_send![self, gpuCoreCount] }
        } else {
            8
        }
    }

    /// Total unified (shared) memory.
    fn shared_memory_size(&self) -> ByteSize {
        if class_responds_to(self, sel!(sharedMemorySize)) {
            ByteSize(unsafe { msg_send![self, sharedMemorySize] })
        } else {
            ByteSize::gib(8)
        }
    }

    /// Whether the GPU supports SIMD group operations.
    fn supports_simd_group(&self) -> bool {
        if class_responds_to(self, sel!(supportsSIMDGroup)) {
            unsafe { msg_send![self, supportsSIMDGroup] }
        } else {
            false
        }
    }

    /// Whether the GPU supports `simdgroup_matrix`.
    fn supports_simd_group_matrix(&self) -> bool {
        if class_responds_to(self, sel!(supportsSIMDGroupMatrix)) {
            unsafe { msg_send![self, supportsSIMDGroupMatrix] }
        } else {
            false
        }
    }

    /// Whether the GPU supports SIMD reduction operations.
    fn supports_simd_reduction(&self) -> bool {
        if class_responds_to(self, sel!(supportsSIMDReduction)) {
            unsafe { msg_send![self, supportsSIMDReduction] }
        } else {
            false
        }
    }

    /// Whether the GPU supports SIMD shuffle-and-fill operations.
    fn supports_simd_shuffle_and_fill(&self) -> bool {
        if class_responds_to(self, sel!(supportsSIMDShuffleAndFill)) {
            unsafe { msg_send![self, supportsSIMDShuffleAndFill] }
        } else {
            false
        }
    }

    /// Whether the GPU supports SIMD shuffles and broadcast.
    fn supports_simd_shuffles_and_broadcast(&self) -> bool {
        if class_responds_to(self, sel!(supportsSIMDShufflesAndBroadcast)) {
            unsafe { msg_send![self, supportsSIMDShufflesAndBroadcast] }
        } else {
            false
        }
    }

    /// Whether the GPU has a Matrix eXtension Unit (neural accelerator for compute).
    /// True on M5+ (Gen18+), false on M1-M4.
    fn supports_mxu(&self) -> bool {
        if class_responds_to(self, sel!(supportsMXU)) {
            unsafe { msg_send![self, supportsMXU] }
        } else {
            false
        }
    }

    /// Whether the GPU supports Thread-Local Storage.
    fn supports_tls(&self) -> bool {
        if class_responds_to(self, sel!(supportsTLS)) {
            unsafe { msg_send![self, supportsTLS] }
        } else {
            false
        }
    }
}

impl DeviceExt for ProtocolObject<dyn MTLDevice> {}
