use crate::backends::{
    common::{Backend, SparseBuffer},
    cpu::Cpu,
};

#[derive(Debug)]
pub struct CpuSparseBuffer {}

impl SparseBuffer for CpuSparseBuffer {
    type Backend = Cpu;

    fn buffer(&self) -> &<Self::Backend as Backend>::Buffer {
        todo!()
    }

    fn buffer_mut(&mut self) -> &mut <Self::Backend as Backend>::Buffer {
        todo!()
    }

    fn capacity(&self) -> usize {
        todo!()
    }

    fn extend(
        &mut self,
        add_length: usize,
    ) {
        todo!()
    }

    fn length(&self) -> usize {
        todo!()
    }
}
