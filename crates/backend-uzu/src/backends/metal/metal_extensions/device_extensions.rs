use core::mem::transmute;
use std::ffi::CString;

use metal::MTLDevice;
use obfstr::obfstr;
use objc2::{
    Message,
    ffi::objc_msgSend,
    runtime::{NSObjectProtocol, ProtocolObject, Sel},
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

// Registers a selector by name at runtime, so the obfuscated private selector
// names are only ever materialized at runtime — never appearing verbatim in the
// binary the way a compile-time `sel!(...)` would.
fn register_selector(name: &str) -> Sel {
    Sel::register(&CString::new(name).expect("selector name has no interior NUL byte"))
}

pub trait DeviceExt: MTLDevice + Message + NSObjectProtocol + Sized {
    /// Number of GPU shader cores.
    fn gpu_core_count(&self) -> u32 {
        let selector = register_selector(obfstr!("gpuCoreCount"));
        if self.respondsToSelector(selector) {
            unsafe { raw_msg_send(self, selector) }
        } else {
            8
        }
    }

    /// Whether the GPU has a Matrix eXtension Unit (neural accelerator for compute).
    /// True on M5+ (Gen18+), false on M1-M4.
    fn supports_mxu(&self) -> bool {
        let selector = register_selector(obfstr!("supportsMXU"));
        if self.respondsToSelector(selector) {
            unsafe { raw_msg_send(self, selector) }
        } else {
            false
        }
    }

    /// Whether the GPU supports placement sparse resources.
    fn supports_placement_sparse_resources(&self) -> bool {
        self.supports_placement_sparse()
    }
}

impl DeviceExt for ProtocolObject<dyn MTLDevice> {}
