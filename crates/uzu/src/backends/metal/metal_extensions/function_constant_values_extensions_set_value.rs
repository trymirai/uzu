use std::ptr::NonNull;

use crate::backends::metal::{MTLDataType, MTLFunctionConstantValues};

/// Extension trait providing ergonomic `set_value` methods for `MTLFunctionConstantValues`.
pub trait FunctionConstantValuesSetValue {
    /// Sets a single value of any type as a function constant.
    fn set_value<T>(
        &self,
        value: &T,
        type_: MTLDataType,
        index: usize,
    );

    /// Sets an array of values of any type as function constants.
    fn set_values<T>(
        &self,
        values: &[T],
        type_: MTLDataType,
        start_index: usize,
    );

    /// Sets a boolean value as a function constant.
    fn set_bool(
        &self,
        value: bool,
        index: usize,
    );

    /// Sets an array of boolean values as function constants.
    fn set_bools(
        &self,
        values: &[bool],
        start_index: usize,
    );

    /// Sets a float value as a function constant.
    fn set_float(
        &self,
        value: f32,
        index: usize,
    );

    /// Sets an array of float values as function constants.
    fn set_floats(
        &self,
        values: &[f32],
        start_index: usize,
    );

    /// Sets an Int32 value as a function constant.
    fn set_int(
        &self,
        value: i32,
        index: usize,
    );

    /// Sets an array of Int32 values as function constants.
    fn set_ints(
        &self,
        values: &[i32],
        start_index: usize,
    );

    /// Sets a UInt32 value as a function constant.
    fn set_uint(
        &self,
        value: u32,
        index: usize,
    );

    /// Sets an array of UInt32 values as function constants.
    fn set_uints(
        &self,
        values: &[u32],
        start_index: usize,
    );
}

impl FunctionConstantValuesSetValue for MTLFunctionConstantValues {
    fn set_value<T>(
        &self,
        value: &T,
        type_: MTLDataType,
        index: usize,
    ) {
        let ptr = NonNull::from(value).cast();
        self.set_constant_value_type_at_index(ptr, type_, index);
    }

    fn set_values<T>(
        &self,
        values: &[T],
        type_: MTLDataType,
        start_index: usize,
    ) {
        if values.is_empty() {
            return;
        }
        let ptr = NonNull::from(&values[0]).cast();
        self.set_constant_values_type_with_range(
            ptr,
            type_,
            start_index..start_index + values.len(),
        );
    }

    fn set_bool(
        &self,
        value: bool,
        index: usize,
    ) {
        self.set_value(&value, MTLDataType::Bool, index);
    }

    fn set_bools(
        &self,
        values: &[bool],
        start_index: usize,
    ) {
        self.set_values(values, MTLDataType::Bool, start_index);
    }

    fn set_float(
        &self,
        value: f32,
        index: usize,
    ) {
        self.set_value(&value, MTLDataType::Float, index);
    }

    fn set_floats(
        &self,
        values: &[f32],
        start_index: usize,
    ) {
        self.set_values(values, MTLDataType::Float, start_index);
    }

    fn set_int(
        &self,
        value: i32,
        index: usize,
    ) {
        self.set_value(&value, MTLDataType::Int, index);
    }

    fn set_ints(
        &self,
        values: &[i32],
        start_index: usize,
    ) {
        self.set_values(values, MTLDataType::Int, start_index);
    }

    fn set_uint(
        &self,
        value: u32,
        index: usize,
    ) {
        self.set_value(&value, MTLDataType::UInt, index);
    }

    fn set_uints(
        &self,
        values: &[u32],
        start_index: usize,
    ) {
        self.set_values(values, MTLDataType::UInt, start_index);
    }
}
