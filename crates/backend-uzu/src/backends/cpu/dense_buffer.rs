use std::{cell::UnsafeCell, os::raw::c_void, pin::Pin, ptr::NonNull};

use super::Cpu;
use crate::backends::common::{Buffer, DenseBuffer};

#[derive(Debug)]
pub struct CpuBuffer(pub(crate) UnsafeCell<Pin<Box<[u8]>>>);

/// SAFETY: contents are accessed through raw pointers with manual
/// synchronization (command submission order, explicit submit/wait).
/// `Send`/`Sync` assert the buffer can be moved and owned across
/// threads, not that individual accesses are race-free.
unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

impl CpuBuffer {
    pub fn new(size: usize) -> Self {
        Self(UnsafeCell::new(Pin::new(vec![0; size].into_boxed_slice())))
    }

    pub(crate) fn get(&self) -> *mut Pin<Box<[u8]>> {
        self.0.get()
    }
}

impl Buffer for CpuBuffer {
    type Backend = Cpu;

    fn gpu_ptr(&self) -> usize {
        unsafe { &*self.0.get() }.as_ptr().addr()
    }

    fn size(&self) -> usize {
        unsafe { &*self.0.get() }.len()
    }
}

impl DenseBuffer for CpuBuffer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { NonNull::new_unchecked((&*self.0.get()).as_ptr() as *mut c_void) }
    }
}
