use std::{cell::UnsafeCell, ops::Range, os::raw::c_void, pin::Pin, ptr::NonNull};

use super::Cpu;
use crate::backends::common::{Backend, Buffer, DenseBuffer};

impl Buffer for UnsafeCell<Pin<Box<[u8]>>> {
    type Backend = Cpu;

    fn as_bytes_slice_range(
        &self,
        _context: Option<&<Self::Backend as Backend>::Context>,
        range: Range<usize>,
    ) -> Result<&[u8], <Self::Backend as Backend>::Error> {
        let pinned = unsafe { &*self.get() };
        Ok(&pinned[range])
    }

    fn gpu_ptr(&self) -> usize {
        unsafe { &*self.get() }.as_ptr().addr()
    }

    fn size(&self) -> usize {
        unsafe { &*self.get() }.len()
    }
}

impl DenseBuffer for UnsafeCell<Pin<Box<[u8]>>> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { NonNull::new_unchecked((&*self.get()).as_ptr() as *mut c_void) }
    }
}
