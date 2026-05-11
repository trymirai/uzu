use std::{ffi::c_void, mem::size_of, ptr::NonNull};

use metal::MTLComputeCommandEncoder;
use objc2::runtime::ProtocolObject;

/// Extension trait providing ergonomic `set_value` methods for compute command encoders.
pub trait ComputeEncoderSetValue {
    /// Sets a value at the specified buffer index.
    fn set_value<T>(
        &self,
        value: &T,
        index: usize,
    );

    /// Sets a slice of values at the specified buffer index.
    fn set_slice<T>(
        &self,
        slice: &[T],
        index: usize,
    );
}

impl ComputeEncoderSetValue for ProtocolObject<dyn MTLComputeCommandEncoder> {
    fn set_value<T>(
        &self,
        value: &T,
        index: usize,
    ) {
        let ptr = NonNull::new(value as *const T as *mut c_void).expect("value pointer should never be null");
        self.set_bytes(ptr, size_of::<T>(), index);
    }

    fn set_slice<T>(
        &self,
        slice: &[T],
        index: usize,
    ) {
        let ptr = NonNull::new(slice.as_ptr() as *mut c_void).expect("slice pointer should never be null");
        self.set_bytes(ptr, size_of::<T>() * slice.len(), index);
    }
}
