pub(crate) const HID_PAGE_APPLE_VENDOR: i32 = 0xff00;
pub(crate) const HID_PAGE_APPLE_VENDOR_POWER: i32 = 0xff08;
pub(crate) const HID_USAGE_TEMPERATURE_SENSOR: i32 = 0x0005;
pub(crate) const HID_USAGE_POWER_VOLTAGE: i32 = 0x0003;
pub(crate) const HID_USAGE_POWER_CURRENT: i32 = 0x0002;

pub(crate) const EVENT_TYPE_TEMPERATURE: i64 = 15;
pub(crate) const EVENT_TYPE_POWER: i64 = 25;

pub(crate) const fn event_field_base(event_type: i64) -> i32 {
    (event_type as i32) << 16
}

#[cfg(target_vendor = "apple")]
mod apple {
    use core::{
        cell::UnsafeCell,
        ffi::c_void,
        marker::{PhantomData, PhantomPinned},
    };
    use std::{ffi::CString, sync::OnceLock};

    use obfstr::obfstr;
    use objc2_core_foundation::{CFAllocator, CFArray, CFDictionary, CFNumber, CFString, CFType};

    // The IOHID handles are CoreFoundation types (they retain/release via
    // `CFRetain`/`CFRelease`). Declaring them as proper CF types — same shape as
    // `objc2-io-surface`'s `IOSurfaceRef` — lets them deref to `CFType` and flow
    // through `CFRetained<T>`, so the FFI below is typed end-to-end instead of
    // trafficking in `*mut c_void`.
    macro_rules! opaque_cf_type {
        ($name:ident) => {
            #[repr(C)]
            #[allow(dead_code)]
            pub(crate) struct $name {
                inner: [u8; 0],
                _p: UnsafeCell<PhantomData<(*const UnsafeCell<()>, PhantomPinned)>>,
            }
            objc2_core_foundation::cf_type!(
                unsafe impl $name {}
            );
        };
    }

    opaque_cf_type!(IOHIDEventSystemClient);
    opaque_cf_type!(IOHIDServiceClient);
    opaque_cf_type!(IOHIDEvent);

    pub(crate) struct Functions {
        pub create: unsafe extern "C" fn(Option<&CFAllocator>) -> *mut IOHIDEventSystemClient,
        pub set_matching: unsafe extern "C" fn(&IOHIDEventSystemClient, &CFDictionary<CFString, CFNumber>) -> i32,
        pub copy_services: unsafe extern "C" fn(&IOHIDEventSystemClient) -> *mut CFArray,
        pub copy_property: unsafe extern "C" fn(&IOHIDServiceClient, &CFString) -> *mut CFType,
        pub copy_event: unsafe extern "C" fn(&IOHIDServiceClient, i64, i32, i64) -> *mut IOHIDEvent,
        pub get_float_value: unsafe extern "C" fn(&IOHIDEvent, i32) -> f64,
        pub get_registry_id: unsafe extern "C" fn(&IOHIDServiceClient) -> u64,
    }

    fn resolve() -> Option<Functions> {
        let path = CString::new(obfstr!("/System/Library/Frameworks/IOKit.framework/IOKit")).ok()?;
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_LAZY) };
        if handle.is_null() {
            return None;
        }

        macro_rules! symbol {
            ($name:literal, $signature:ty) => {{
                let name = CString::new(obfstr!($name)).ok()?;
                let symbol = unsafe { libc::dlsym(handle, name.as_ptr()) };
                if symbol.is_null() {
                    return None;
                }
                unsafe { core::mem::transmute::<*mut c_void, $signature>(symbol) }
            }};
        }

        Some(Functions {
            create: symbol!(
                "IOHIDEventSystemClientCreate",
                unsafe extern "C" fn(Option<&CFAllocator>) -> *mut IOHIDEventSystemClient
            ),
            set_matching: symbol!(
                "IOHIDEventSystemClientSetMatching",
                unsafe extern "C" fn(&IOHIDEventSystemClient, &CFDictionary<CFString, CFNumber>) -> i32
            ),
            copy_services: symbol!(
                "IOHIDEventSystemClientCopyServices",
                unsafe extern "C" fn(&IOHIDEventSystemClient) -> *mut CFArray
            ),
            copy_property: symbol!(
                "IOHIDServiceClientCopyProperty",
                unsafe extern "C" fn(&IOHIDServiceClient, &CFString) -> *mut CFType
            ),
            copy_event: symbol!(
                "IOHIDServiceClientCopyEvent",
                unsafe extern "C" fn(&IOHIDServiceClient, i64, i32, i64) -> *mut IOHIDEvent
            ),
            get_float_value: symbol!("IOHIDEventGetFloatValue", unsafe extern "C" fn(&IOHIDEvent, i32) -> f64),
            get_registry_id: symbol!(
                "IOHIDServiceClientGetRegistryID",
                unsafe extern "C" fn(&IOHIDServiceClient) -> u64
            ),
        })
    }

    pub(crate) fn functions() -> Option<&'static Functions> {
        static FUNCTIONS: OnceLock<Option<Functions>> = OnceLock::new();
        FUNCTIONS.get_or_init(resolve).as_ref()
    }
}

#[cfg(target_vendor = "apple")]
pub(crate) use apple::{Functions, IOHIDEvent, IOHIDEventSystemClient, IOHIDServiceClient, functions};
