use std::convert::Infallible;

use crate::backends::{
    common::{Backend, Buffer, GpuBuffer, SparseBuffer},
    cpu::Cpu,
};

#[derive(Debug)]
pub struct CpuSparseBuffer {
    never: Infallible,
}

impl Buffer for CpuSparseBuffer {
    type Backend = Cpu;

    fn size(&self) -> usize {
        match self.never {}
    }
}

impl GpuBuffer for CpuSparseBuffer {}

impl SparseBuffer for CpuSparseBuffer {
    fn map(
        &mut self,
        _until: usize,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        match self.never {}
    }
}
