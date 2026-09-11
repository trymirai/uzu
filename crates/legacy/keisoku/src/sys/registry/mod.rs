use core::{ffi::c_void, ptr::NonNull};

use objc2_core_foundation::{CFData, CFDictionary, CFRetained, CFString, CFType, ConcreteType};

mod service_iterator;

pub use service_iterator::IoServiceIterator;

pub fn cf_string_to_string(value: *const c_void) -> String {
    match NonNull::new(value.cast_mut().cast::<CFString>()) {
        Some(pointer) => unsafe { pointer.as_ref() }.to_string(),
        None => String::new(),
    }
}

pub fn dictionary_get<V: ConcreteType>(
    dictionary: &CFDictionary,
    key: &str,
) -> Option<CFRetained<V>> {
    let dictionary: &CFDictionary<CFString, CFType> = unsafe { &*(dictionary as *const CFDictionary).cast() };
    dictionary.get(&CFString::from_str(key)).and_then(|value| value.downcast::<V>().ok())
}

pub fn dictionary_data(
    dictionary: &CFDictionary,
    key: &str,
) -> Option<Box<[u8]>> {
    dictionary_get::<CFData>(dictionary, key).map(|data| cf_data_to_bytes(&data))
}

fn cf_data_to_bytes(data: &CFData) -> Box<[u8]> {
    let length = data.length();
    let bytes = data.byte_ptr();
    if length <= 0 || bytes.is_null() {
        return Box::default();
    }
    unsafe { core::slice::from_raw_parts(bytes, length as usize) }.into()
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
