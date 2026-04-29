use std::{cell::UnsafeCell, pin::Pin};

use backend_uzu::backends::common::SparsePagesOperation;

use crate::backends::{common::SparsePages, cpu::Cpu};

#[derive(Debug)]
pub struct CpuSparsePages {
    buffer: UnsafeCell<Pin<Box<[u8]>>>,
}

impl CpuSparsePages {
    pub fn new() -> Self {
        Self {
            buffer: UnsafeCell::new(Pin::new(vec![0; 0].into_boxed_slice())),
        }
    }
}

impl SparsePages for CpuSparsePages {
    type Backend = Cpu;

    fn buffer(&self) -> &UnsafeCell<Pin<Box<[u8]>>> {
        &self.buffer
    }

    fn buffer_mut(&mut self) -> &mut UnsafeCell<Pin<Box<[u8]>>> {
        &mut self.buffer
    }

    fn execute(
        &mut self,
        _operations: &[SparsePagesOperation],
    ) {
        unimplemented!("CPU backend does not support sparse pages");
    }

    fn page_size(&self) -> usize {
        unimplemented!("CPU backend does not support sparse pages");
    }

    fn total_pages(&self) -> usize {
        unimplemented!("CPU backend does not support sparse pages");
    }
}
