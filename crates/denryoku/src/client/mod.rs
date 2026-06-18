use objc2_core_foundation::{CFAllocator, CFArray, CFDictionary, CFNumber, CFString, CFType};

kanka::opaque_cf_type!(IOHIDEventSystemClient);
kanka::opaque_cf_type!(IOHIDServiceClient);
kanka::opaque_cf_type!(IOHIDEvent);

kanka::ffi_table! {
    struct IOKit from "/System/Library/Frameworks/IOKit.framework/IOKit" {
        create = "IOHIDEventSystemClientCreate":
            unsafe extern "C" fn(Option<&CFAllocator>) -> *mut IOHIDEventSystemClient,
        set_matching = "IOHIDEventSystemClientSetMatching":
            unsafe extern "C" fn(&IOHIDEventSystemClient, &CFDictionary<CFString, CFNumber>) -> i32,
        copy_services = "IOHIDEventSystemClientCopyServices":
            unsafe extern "C" fn(&IOHIDEventSystemClient) -> *mut CFArray,
        copy_property = "IOHIDServiceClientCopyProperty":
            unsafe extern "C" fn(&IOHIDServiceClient, &CFString) -> *mut CFType,
        copy_event = "IOHIDServiceClientCopyEvent":
            unsafe extern "C" fn(&IOHIDServiceClient, i64, i32, i64) -> *mut IOHIDEvent,
        get_float_value = "IOHIDEventGetFloatValue":
            unsafe extern "C" fn(&IOHIDEvent, i32) -> f64,
        get_registry_id = "IOHIDServiceClientGetRegistryID":
            unsafe extern "C" fn(&IOHIDServiceClient) -> u64,
    }
}

mod event_system_client;
mod sampler;
mod service_client;

pub use sampler::Sampler;
pub(crate) use sampler::collect;

/// Whether the private IOKit HID API resolved on this system.
pub(crate) fn is_available() -> bool {
    IOKit::get().is_some()
}
