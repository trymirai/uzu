//! Thin CoreFoundation/IOKit helpers built on the typed `objc2-core-foundation`
//! and `objc2-io-kit` bindings — no hand-rolled `extern` blocks.
//!
//! The IOReport accessors hand back raw `*const c_void` (they're private symbols
//! bound through `kanka`), so these helpers bridge between those raw pointers and
//! the typed CF wrappers. `CFRetained<T>` gives RAII release for owned values.

use core::{ffi::c_void, ptr::NonNull};

use objc2_core_foundation::{CFData, CFDictionary, CFRetained, CFString};

mod service_iterator;

pub use service_iterator::IoServiceIterator;

/// Reads a `CFString` (by raw pointer, e.g. from an IOReport accessor) into a
/// Rust `String`. Empty on null.
pub fn cf_string_to_string(value: *const c_void) -> String {
    match NonNull::new(value.cast_mut().cast::<CFString>()) {
        Some(pointer) => unsafe { pointer.as_ref() }.to_string(),
        None => String::new(),
    }
}

/// `dictionary[key]` (borrowed, not retained); `None` if absent.
pub fn cf_dictionary_value(
    dictionary: &CFDictionary,
    key: &str,
) -> Option<*const c_void> {
    let key = CFString::from_str(key);
    let value = unsafe { dictionary.value(CFRetained::as_ptr(&key).as_ptr() as *const c_void) };
    (!value.is_null()).then_some(value)
}

/// Copies a `CFData` value's bytes (by raw pointer) into a `Vec`; empty if the
/// pointer is null or the data is empty.
pub fn cf_data_bytes(value: *const c_void) -> Vec<u8> {
    let Some(pointer) = NonNull::new(value.cast_mut().cast::<CFData>()) else {
        return Vec::new();
    };
    let data = unsafe { pointer.as_ref() };
    let length = data.length();
    let bytes = data.byte_ptr();
    if length <= 0 || bytes.is_null() {
        return Vec::new();
    }
    unsafe { core::slice::from_raw_parts(bytes, length as usize) }.to_vec()
}

/// Reads an IORegistry entry's CF property dictionary (released on drop).
pub fn registry_properties(entry: objc2_io_kit::io_registry_entry_t) -> Option<CFRetained<CFDictionary>> {
    let mut properties = core::ptr::null_mut();
    let result = unsafe { objc2_io_kit::IORegistryEntryCreateCFProperties(entry, &mut properties, None, 0) };
    if result != 0 {
        return None;
    }
    let mutable = unsafe { CFRetained::from_raw(NonNull::new(properties)?) };
    Some(unsafe { CFRetained::from_raw(CFRetained::into_raw(mutable).cast()) })
}
