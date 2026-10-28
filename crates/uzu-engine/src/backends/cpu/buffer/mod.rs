use std::{os::raw::c_void, ptr::NonNull};

use crate::{
    backends::{
        common::{Backend, CpuBuffer, GpuBuffer},
        cpu::Cpu,
    },
    utils::downcast::downcast_ref,
};

pub mod dense;
pub mod sparse;

pub(super) trait CpuBufferExt: GpuBuffer<Backend = Cpu> {
    fn cpu_address(&self) -> NonNull<c_void> {
        if let Some(buffer) = downcast_ref::<<Cpu as Backend>::GlobalGpuBuffer>(self) {
            buffer.cpu_ptr()
        } else if let Some(buffer) = downcast_ref::<<Cpu as Backend>::ConstantBuffer>(self) {
            buffer.cpu_ptr()
        } else if let Some(buffer) = downcast_ref::<<Cpu as Backend>::ScratchBuffer>(self) {
            buffer.cpu_ptr()
        } else {
            unreachable!("Unsupported Cpu buffer type")
        }
    }
}

impl<T: GpuBuffer<Backend = Cpu> + ?Sized> CpuBufferExt for T {}
