use bytesize::ByteSize;
use metal::MTLDevice;
use objc2::{
    Message, msg_send,
    rc::Retained,
    runtime::{NSObjectProtocol, ProtocolObject},
    sel,
};
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

pub trait DeviceExt: MTLDevice + Message + NSObjectProtocol + Sized {
    fn family_name(&self) -> String {
        if self.respondsToSelector(sel!(familyName)) {
            let ns: Retained<NSString> = unsafe { msg_send![self, familyName] };
            ns.to_string()
        } else {
            "Unknown".into()
        }
    }

    fn gpu_core_count(&self) -> u32 {
        if self.respondsToSelector(sel!(gpuCoreCount)) {
            unsafe { msg_send![self, gpuCoreCount] }
        } else {
            8
        }
    }

    fn shared_memory_size(&self) -> ByteSize {
        if self.respondsToSelector(sel!(sharedMemorySize)) {
            ByteSize(unsafe { msg_send![self, sharedMemorySize] })
        } else {
            ByteSize::gib(8)
        }
    }

    fn supports_simd_group(&self) -> bool {
        if self.respondsToSelector(sel!(supportsSIMDGroup)) {
            unsafe { msg_send![self, supportsSIMDGroup] }
        } else {
            false
        }
    }

    fn supports_simd_group_matrix(&self) -> bool {
        if self.respondsToSelector(sel!(supportsSIMDGroupMatrix)) {
            unsafe { msg_send![self, supportsSIMDGroupMatrix] }
        } else {
            false
        }
    }

    fn supports_simd_reduction(&self) -> bool {
        if self.respondsToSelector(sel!(supportsSIMDReduction)) {
            unsafe { msg_send![self, supportsSIMDReduction] }
        } else {
            false
        }
    }

    fn supports_simd_shuffle_and_fill(&self) -> bool {
        if self.respondsToSelector(sel!(supportsSIMDShuffleAndFill)) {
            unsafe { msg_send![self, supportsSIMDShuffleAndFill] }
        } else {
            false
        }
    }

    fn supports_simd_shuffles_and_broadcast(&self) -> bool {
        if self.respondsToSelector(sel!(supportsSIMDShufflesAndBroadcast)) {
            unsafe { msg_send![self, supportsSIMDShufflesAndBroadcast] }
        } else {
            false
        }
    }

    fn supports_mxu(&self) -> bool {
        if self.respondsToSelector(sel!(supportsMXU)) {
            unsafe { msg_send![self, supportsMXU] }
        } else {
            false
        }
    }

    fn supports_tls(&self) -> bool {
        if self.respondsToSelector(sel!(supportsTLS)) {
            unsafe { msg_send![self, supportsTLS] }
        } else {
            false
        }
    }
}

impl DeviceExt for ProtocolObject<dyn MTLDevice> {}
