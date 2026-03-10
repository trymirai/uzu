use std::{os::raw::c_void, ptr::NonNull};

use super::Cpu;
use crate::backends::common::Buffer;

impl Buffer for Box<[u8]> {
    type Backend = Cpu;

    fn set_label(
        &mut self,
        _label: Option<&str>,
    ) {
    }

    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { NonNull::new_unchecked(self.as_ptr() as *mut c_void) }
    }
    fn length(&self) -> usize {
        self.len()
    }
    fn id(&self) -> usize {
        unimplemented!()
    }
}
