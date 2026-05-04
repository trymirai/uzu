use core::mem::transmute;

use bytesize::ByteSize;
use metal::prelude::*;
use objc2::{
    Message,
    ffi::objc_msgSend,
    msg_send,
    runtime::{NSObjectProtocol, Sel},
    sel,
};
use objc2_foundation::NSString;

// Used to bypass objc2's debug-mode class validation for proxy objects like
// `CaptureMTLDevice` which forward messages at the ObjC runtime level (via
// `forwardingTargetForSelector:`) but don't have the methods in their own class
// table — causing `msg_send!`'s debug check to panic even though the message
// would succeed.
//
// The caller must verify with `respondsToSelector:` first.
unsafe fn raw_msg_send<T: Message + ?Sized, R>(
    obj: &T,
    sel: Sel,
) -> R {
    let object_pointer: *const T = obj;
    let send: unsafe extern "C" fn(*const T, Sel) -> R = unsafe { transmute(objc_msgSend as *const ()) };
    unsafe { send(object_pointer, sel) }
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

pub trait DeviceExt: MTLDevice + Message + NSObjectProtocol + Sized {
    /// Human-readable chip name, e.g. "M2 Max", "M4 Pro", "A17 Pro".
    fn family_name(&self) -> String {
        if self.respondsToSelector(sel!(familyName)) {
            let family_name: Retained<NSString> = unsafe { msg_send![self, familyName] };
            family_name.to_string()
        } else {
            "Unknown".into()
        }
    }

    /// Number of GPU shader cores.
    fn gpu_core_count(&self) -> u32 {
        if self.respondsToSelector(sel!(gpuCoreCount)) {
            unsafe { raw_msg_send(self, sel!(gpuCoreCount)) }
        } else {
            8
        }
    }

    /// Total unified (shared) memory.
    fn shared_memory_size(&self) -> ByteSize {
        if self.respondsToSelector(sel!(sharedMemorySize)) {
            ByteSize(unsafe { raw_msg_send(self, sel!(sharedMemorySize)) })
        } else {
            ByteSize::gib(8)
        }
    }

    /// Whether the GPU supports SIMD group operations.
    fn supports_simd_group(&self) -> bool {
        if self.respondsToSelector(sel!(supportsSIMDGroup)) {
            unsafe { raw_msg_send(self, sel!(supportsSIMDGroup)) }
        } else {
            false
        }
    }

    /// Whether the GPU supports `simdgroup_matrix`.
    fn supports_simd_group_matrix(&self) -> bool {
        if self.respondsToSelector(sel!(supportsSIMDGroupMatrix)) {
            unsafe { raw_msg_send(self, sel!(supportsSIMDGroupMatrix)) }
        } else {
            false
        }
    }

    /// Whether the GPU supports SIMD reduction operations.
    fn supports_simd_reduction(&self) -> bool {
        if self.respondsToSelector(sel!(supportsSIMDReduction)) {
            unsafe { raw_msg_send(self, sel!(supportsSIMDReduction)) }
        } else {
            false
        }
    }

    /// Whether the GPU supports SIMD shuffle-and-fill operations.
    fn supports_simd_shuffle_and_fill(&self) -> bool {
        if self.respondsToSelector(sel!(supportsSIMDShuffleAndFill)) {
            unsafe { raw_msg_send(self, sel!(supportsSIMDShuffleAndFill)) }
        } else {
            false
        }
    }

    /// Whether the GPU supports SIMD shuffles and broadcast.
    fn supports_simd_shuffles_and_broadcast(&self) -> bool {
        if self.respondsToSelector(sel!(supportsSIMDShufflesAndBroadcast)) {
            unsafe { raw_msg_send(self, sel!(supportsSIMDShufflesAndBroadcast)) }
        } else {
            false
        }
    }

    /// Whether the GPU has a Matrix eXtension Unit (neural accelerator for compute).
    /// True on M5+ (Gen18+), false on M1-M4.
    fn supports_mxu(&self) -> bool {
        if self.respondsToSelector(sel!(supportsMXU)) {
            unsafe { raw_msg_send(self, sel!(supportsMXU)) }
        } else {
            false
        }
    }

    /// Whether the GPU supports Thread-Local Storage.
    fn supports_tls(&self) -> bool {
        if self.respondsToSelector(sel!(supportsTLS)) {
            unsafe { raw_msg_send(self, sel!(supportsTLS)) }
        } else {
            false
        }
    }
}

impl DeviceExt for ProtocolObject<dyn MTLDevice> {}
