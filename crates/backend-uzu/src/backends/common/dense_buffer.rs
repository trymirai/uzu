use std::{os::raw::c_void, ptr::NonNull};

use super::{Backend, Buffer};

pub trait DenseBuffer: Buffer<Backend: Backend<DenseBuffer = Self>> {
    fn cpu_ptr(&self) -> NonNull<c_void>;
}
