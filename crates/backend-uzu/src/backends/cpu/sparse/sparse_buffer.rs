use std::ops::Range;

use crate::backends::{
    common::{Backend, Buffer, SparseBuffer},
    cpu::Cpu,
};

#[derive(Debug)]
pub struct CpuSparseBuffer {}

impl Buffer for CpuSparseBuffer {
    type Backend = Cpu;

    fn gpu_ptr(&self) -> usize {
        todo!()
    }

    fn size(&self) -> usize {
        todo!()
    }
}

impl SparseBuffer for CpuSparseBuffer {
    fn map(
        &mut self,
        _context: &<Self::Backend as Backend>::Context,
        _pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn unmap(
        &mut self,
        _context: &<Self::Backend as Backend>::Context,
        _pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        todo!()
    }
}
