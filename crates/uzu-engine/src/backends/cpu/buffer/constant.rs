use std::{fmt::Debug, os::raw::c_void, ptr::NonNull};

use crate::backends::{
    common::{Backend, Buffer, BufferCpuAccessible, ConstantBuffer, allocator::bump::BumpAllocation},
    cpu::Cpu,
};

impl Buffer for BumpAllocation<<Cpu as Backend>::GlobalBuffer> {
    type Backend = Cpu;

    fn size(&self) -> usize {
        self.range().iter().len()
    }
}

impl BufferCpuAccessible for BumpAllocation<<Cpu as Backend>::GlobalBuffer> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { self.page().cpu_ptr().byte_add(self.range().start) }
    }
}

impl ConstantBuffer for BumpAllocation<<Cpu as Backend>::GlobalBuffer> {}

impl Debug for BumpAllocation<<Cpu as Backend>::GlobalBuffer> {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("BumpAllocation<Cpu::GlobalBuffer>")
            .field("page", &self.page())
            .field("range", &self.range())
            .finish_non_exhaustive()
    }
}
