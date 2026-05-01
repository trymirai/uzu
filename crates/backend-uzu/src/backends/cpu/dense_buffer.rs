use std::{cell::UnsafeCell, os::raw::c_void, pin::Pin, ptr::NonNull};

use super::Cpu;
use crate::backends::common::{Buffer, DenseBuffer};

impl Buffer for UnsafeCell<Pin<Box<[u8]>>> {
    fn gpu_ptr(&self) -> usize {
        unsafe { &*self.get() }.as_ptr().addr()
    }

    fn size(&self) -> usize {
        unsafe { &*self.get() }.len()
    }

    fn set_label(
        &mut self,
        _label: Option<&str>,
    ) {
    }
}

impl DenseBuffer for UnsafeCell<Pin<Box<[u8]>>> {
    type Backend = Cpu;

    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { NonNull::new_unchecked((&*self.get()).as_ptr() as *mut c_void) }
    }
}
