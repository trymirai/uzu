use std::{cell::UnsafeCell, os::raw::c_void, pin::Pin, ptr::NonNull};

use crate::backends::{
    common::{
        BufferCpuAccessible, ConstantBuffer, GlobalBuffer, ScratchBuffer,
        allocator::{Allocation, Storage},
    },
    cpu::Cpu,
};

#[derive(Debug)]
pub struct CpuBuffer(UnsafeCell<Pin<Box<[u8]>>>);

/// SAFETY: contents are accessed through raw pointers with manual
/// synchronization (command submission order, explicit submit/wait).
/// `Send`/`Sync` assert the buffer can be moved and owned across
/// threads, not that individual accesses are race-free.
unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

impl CpuBuffer {
    pub(in crate::backends::cpu) fn new(size: usize) -> Self {
        Self(UnsafeCell::new(Pin::new(vec![0; size].into_boxed_slice())))
    }
}

impl Storage for CpuBuffer {
    type Backend = Cpu;

    const MIN_ALLOCATION_ALIGNMENT: usize = 4;
    const MAX_ALLOCATION_ALIGNMENT: usize = 64;
    const ALLOCATION_GRANULARITY: usize = 8 * 1024 * 1024;
}

impl BufferCpuAccessible for Allocation<CpuBuffer> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { NonNull::new_unchecked((&*self.buffer().0.get()).as_ptr().add(self.offset()) as *mut c_void) }
    }
}

impl GlobalBuffer for Allocation<CpuBuffer> {}
impl ConstantBuffer for Allocation<CpuBuffer> {}
impl ScratchBuffer for Allocation<CpuBuffer> {}
