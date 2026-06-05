use core::mem::transmute;

use metal::MTLDevice;
use objc2::{
    Message,
    ffi::objc_msgSend,
    runtime::{NSObjectProtocol, ProtocolObject, Sel},
    sel,
};

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

pub trait DeviceExt: MTLDevice + Message + NSObjectProtocol + Sized {
    /// Number of GPU shader cores.
    fn gpu_core_count(&self) -> u32 {
        if self.respondsToSelector(sel!(gpuCoreCount)) {
            unsafe { raw_msg_send(self, sel!(gpuCoreCount)) }
        } else {
            8
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

    /// Whether the GPU supports placement sparse resources.
    fn supports_placement_sparse_resources(&self) -> bool {
        if self.respondsToSelector(sel!(supportsPlacementSparse)) {
            unsafe { raw_msg_send(self, sel!(supportsPlacementSparse)) }
        } else {
            false
        }
    }
}

impl DeviceExt for ProtocolObject<dyn MTLDevice> {}
