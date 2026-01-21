use std::ptr::NonNull;

use crate::backends::metal::{MTLDataType, MTLFunctionConstantValues};

/// Extension trait providing legacy-style API for function constant values.
/// This matches the old metal-rs API signature to minimize migration changes.
pub trait FunctionConstantValuesLegacy {
    /// Set constant value at index with legacy parameter order.
    /// Old API: set_constant_value_at_index(ptr, dtype, index)
    /// New API: set_constant_value_type_at_index(NonNull, dtype, index)
    fn set_constant_value_at_index(
        &self,
        value: *const std::ffi::c_void,
        data_type: MTLDataType,
        index: u64,
    );
}

impl FunctionConstantValuesLegacy for MTLFunctionConstantValues {
    fn set_constant_value_at_index(
        &self,
        value: *const std::ffi::c_void,
        data_type: MTLDataType,
        index: u64,
    ) {
        if let Some(ptr) = NonNull::new(value as *mut std::ffi::c_void) {
            unsafe {
                MTLFunctionConstantValues::set_constant_value_type_at_index(
                    self,
                    ptr,
                    data_type,
                    index as usize,
                );
            }
        }
    }
}
