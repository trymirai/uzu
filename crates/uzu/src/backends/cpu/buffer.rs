use std::{os::raw::c_void, ptr::NonNull};

use crate::backends::{common::NativeBuffer, cpu::backend::Cpu};

#[derive(Debug, Clone)]
pub struct CpuBuffer;

impl NativeBuffer for CpuBuffer {
    type Backend = Cpu;

    fn set_label(
        &self,
        label: Option<&str>,
    ) {
        todo!()
    }

    fn cpu_ptr(&self) -> NonNull<c_void> {
        todo!()
    }

    fn length(&self) -> usize {
        todo!()
    }

    fn id(&self) -> usize {
        todo!()
    }
}
