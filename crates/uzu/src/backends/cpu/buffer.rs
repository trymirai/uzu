use std::{os::raw::c_void, ptr::NonNull, sync::Arc};

use crate::backends::{common::NativeBuffer, cpu::backend::Cpu};

pub type CpuBuffer = Arc<Box<[u8]>>;

impl NativeBuffer for CpuBuffer {
    type Backend = Cpu;

    fn set_label(
        &self,
        _label: Option<&str>,
    ) {
    }

    fn cpu_ptr(&self) -> NonNull<c_void> {
        let mut_ptr = self.as_ptr() as *mut c_void;
        NonNull::new(mut_ptr).unwrap()
    }

    fn length(&self) -> usize {
        self.len()
    }

    fn id(&self) -> usize {
        self.as_ptr() as usize
    }
}
