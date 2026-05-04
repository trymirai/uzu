use rangemap::RangeMap;

use crate::backends::{
    common::{Backend, SparseBuffer, SparseBufferOperation},
    cpu::Cpu,
};

#[derive(Debug)]
pub struct CpuSparseBuffer {}

impl SparseBuffer for CpuSparseBuffer {
    type Backend = Cpu;

    fn set_label(
        &mut self,
        _label: Option<&str>,
    ) {
        todo!()
    }

    fn gpu_ptr(&self) -> usize {
        todo!()
    }

    fn length(&self) -> usize {
        todo!()
    }

    fn get_mapped_pages(&self) -> &RangeMap<usize, ()> {
        todo!()
    }

    fn get_page_size(&self) -> usize {
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
