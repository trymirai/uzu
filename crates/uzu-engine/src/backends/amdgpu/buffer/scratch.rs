use crate::backends::{
    amdgpu::{Amdgpu, buffer::device::ScratchPage},
    common::{Buffer, ScratchBuffer, allocator::pool::PoolAllocation},
};

impl Buffer for PoolAllocation<ScratchPage> {
    type Backend = Amdgpu;

    fn size(&self) -> usize {
        self.range().iter().len()
    }
}

impl ScratchBuffer for PoolAllocation<ScratchPage> {}
