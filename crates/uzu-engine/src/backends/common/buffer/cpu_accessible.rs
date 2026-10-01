use std::{os::raw::c_void, ptr::NonNull};

use crate::backends::common::Buffer;

pub trait BufferCpuAccessible: Buffer {
    fn cpu_ptr(&self) -> NonNull<c_void>;
}
