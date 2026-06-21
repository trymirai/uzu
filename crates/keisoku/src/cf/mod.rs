use core::{ffi::c_void, ptr::NonNull};

use objc2_core_foundation::{CFData, CFDictionary, CFRetained, CFString, CFType, ConcreteType};

mod service_iterator;

pub use service_iterator::IoServiceIterator;

/// Convert a borrowed (Get-rule) `CFStringRef` returned by a C accessor into an
/// owned `String`. Used by the IOReport channel accessors, whose C functions
/// hand back a borrowed `CFStringRef` as `*const c_void`.
pub fn cf_string_to_string(value: *const c_void) -> String {
    match NonNull::new(value.cast_mut().cast::<CFString>()) {
        Some(pointer) => unsafe { pointer.as_ref() }.to_string(),
        None => String::new(),
    }
}

/// Look up `key` in a CF dictionary and downcast its value to the concrete CF
/// type `V`. The `<K, V>` generics on `CFDictionary` are `PhantomData`, so
/// viewing the opaque dictionary as `<CFString, CFType>` is a layout-sound
/// reference cast.
pub fn dictionary_get<V: ConcreteType>(
    dictionary: &CFDictionary,
    key: &str,
) -> Option<CFRetained<V>> {
    let dictionary: &CFDictionary<CFString, CFType> = unsafe { &*(dictionary as *const CFDictionary).cast() };
    dictionary.get(&CFString::from_str(key)).and_then(|value| value.downcast::<V>().ok())
}

/// Bytes of a `CFData`-valued key, or `None` if the key is absent or not data.
pub fn dictionary_data(
    dictionary: &CFDictionary,
    key: &str,
) -> Option<Vec<u8>> {
    dictionary_get::<CFData>(dictionary, key).map(|data| cf_data_to_vec(&data))
}

fn cf_data_to_vec(data: &CFData) -> Vec<u8> {
    let length = data.length();
    let bytes = data.byte_ptr();
    if length <= 0 || bytes.is_null() {
        return Vec::new();
    }
    unsafe { core::slice::from_raw_parts(bytes, length as usize) }.to_vec()
}

pub fn registry_properties(entry: objc2_io_kit::io_registry_entry_t) -> Option<CFRetained<CFDictionary>> {
    let mut properties = core::ptr::null_mut();
    let result = unsafe { objc2_io_kit::IORegistryEntryCreateCFProperties(entry, &mut properties, None, 0) };
    if result != 0 {
        return None;
    }
    let mutable = unsafe { CFRetained::from_raw(NonNull::new(properties)?) };
    Some(unsafe { CFRetained::from_raw(CFRetained::into_raw(mutable).cast()) })
}
