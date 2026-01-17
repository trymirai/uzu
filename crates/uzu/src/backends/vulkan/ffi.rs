use std::ffi::{c_char, CStr, CString};

pub fn c_char_slice_to_string(raw_string_array: &[c_char]) -> String {
    unsafe { CStr::from_ptr(raw_string_array.as_ptr()) }
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}

pub fn str_to_ptr_const_char(input: &str) -> *const c_char {
    CString::new(input).unwrap().into_raw()
}