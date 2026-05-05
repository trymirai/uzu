use bytesize::ByteSize;
use rangemap::RangeMap;

use crate::backends::{
    common::{Backend, Buffer, SparseBuffer, SparseBufferOperation},
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
    fn get_mapped_pages(&self) -> &RangeMap<usize, ()> {
        todo!()
    }

    fn get_page_size(&self) -> ByteSize {
        todo!()
    }

    fn execute(
        &mut self,
        _context: &<Self::Backend as Backend>::Context,
        _operations: &[SparseBufferOperation],
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        todo!()
    }
}
