use std::{cell::UnsafeCell, os::raw::c_void, pin::Pin, ptr::NonNull};

use super::Cpu;
use crate::backends::common::Buffer;

impl Buffer for UnsafeCell<Pin<Box<[u8]>>> {
    type Backend = Cpu;

    fn set_label(
        &mut self,
        _label: Option<&str>,
    ) {
    }

    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { NonNull::new_unchecked((&*self.get()).as_ptr() as *mut c_void) }
    }

    fn gpu_ptr(&self) -> usize {
        unsafe { &*self.get() }.as_ptr().addr()
    }

    fn length(&self) -> usize {
        unsafe { &*self.get() }.len()
    }
}
