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
    use core::ffi::{c_char, c_void};
    use std::sync::OnceLock;

    pub(crate) struct Functions {
        pub create: unsafe extern "C" fn(*const c_void) -> *mut c_void,
        pub set_matching: unsafe extern "C" fn(*mut c_void, *const c_void) -> i32,
        pub copy_services: unsafe extern "C" fn(*mut c_void) -> *const c_void,
        pub copy_property: unsafe extern "C" fn(*mut c_void, *const c_void) -> *const c_void,
        pub copy_event: unsafe extern "C" fn(*mut c_void, i64, i32, i64) -> *mut c_void,
        pub get_float_value: unsafe extern "C" fn(*mut c_void, i32) -> f64,
        pub get_registry_id: unsafe extern "C" fn(*mut c_void) -> u64,
    }

    const IOKIT_FRAMEWORK_PATH: &[u8] = b"/System/Library/Frameworks/IOKit.framework/IOKit\0";

    fn resolve() -> Option<Functions> {
        let handle = unsafe { libc::dlopen(IOKIT_FRAMEWORK_PATH.as_ptr().cast::<c_char>(), libc::RTLD_LAZY) };
        if handle.is_null() {
            return None;
        }

        macro_rules! symbol {
            ($name:literal, $signature:ty) => {{
                let symbol = unsafe { libc::dlsym(handle, concat!($name, "\0").as_ptr().cast::<c_char>()) };
                if symbol.is_null() {
                    return None;
                }
                unsafe { core::mem::transmute::<*mut c_void, $signature>(symbol) }
            }};
        }

        Some(Functions {
            create: symbol!("IOHIDEventSystemClientCreate", unsafe extern "C" fn(*const c_void) -> *mut c_void),
            set_matching: symbol!(
                "IOHIDEventSystemClientSetMatching",
                unsafe extern "C" fn(*mut c_void, *const c_void) -> i32
            ),
            copy_services: symbol!(
                "IOHIDEventSystemClientCopyServices",
                unsafe extern "C" fn(*mut c_void) -> *const c_void
            ),
            copy_property: symbol!(
                "IOHIDServiceClientCopyProperty",
                unsafe extern "C" fn(*mut c_void, *const c_void) -> *const c_void
            ),
            copy_event: symbol!(
                "IOHIDServiceClientCopyEvent",
                unsafe extern "C" fn(*mut c_void, i64, i32, i64) -> *mut c_void
            ),
            get_float_value: symbol!("IOHIDEventGetFloatValue", unsafe extern "C" fn(*mut c_void, i32) -> f64),
            get_registry_id: symbol!("IOHIDServiceClientGetRegistryID", unsafe extern "C" fn(*mut c_void) -> u64),
        })
    }

    pub(crate) fn functions() -> Option<&'static Functions> {
        static FUNCTIONS: OnceLock<Option<Functions>> = OnceLock::new();
        FUNCTIONS.get_or_init(resolve).as_ref()
    }
}

#[cfg(target_vendor = "apple")]
pub(crate) use apple::{Functions, functions};
