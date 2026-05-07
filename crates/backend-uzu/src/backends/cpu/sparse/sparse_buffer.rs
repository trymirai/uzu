use std::ops::Range;

use bytesize::ByteSize;

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

    fn size(&self) -> ByteSize {
        todo!()
    }

    fn set_label(
        &mut self,
        _label: Option<&str>,
    ) {
        todo!()
    }
}

impl SparseBuffer for CpuSparseBuffer {
    fn mapping(
        &mut self,
        _context: &<Self::Backend as Backend>::Context,
        _pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        todo!()
    }

    fn unmapping(
        &mut self,
        _context: &<Self::Backend as Backend>::Context,
        _pages: &Range<usize>,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        todo!()
    }
}
