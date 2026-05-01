use std::{cell::UnsafeCell, pin::Pin};

use backend_uzu::backends::common::{Backend, Buffer, SparseBufferOperation};

use crate::backends::{
    common::SparseBuffer,
    cpu::{Cpu, error::CpuError},
};

#[derive(Debug)]
pub struct CpuSparseBuffer {
    buffer: UnsafeCell<Pin<Box<[u8]>>>,
}

impl CpuSparseBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: UnsafeCell::new(Pin::new(vec![0; capacity].into_boxed_slice())),
        }
    }
}

impl SparseBuffer for CpuSparseBuffer {
    type Backend = Cpu;

    fn buffer(&self) -> &<Self::Backend as Backend>::Buffer {
        &self.buffer
    }

    fn buffer_mut(&mut self) -> &mut <Self::Backend as Backend>::Buffer {
        &mut self.buffer
    }

    fn set_label(
        &mut self,
        _label: Option<&str>,
    ) {
        self.buffer.set_label(_label)
    }

    fn gpu_ptr(&self) -> usize {
        self.buffer.gpu_ptr()
    }

    fn length(&self) -> usize {
        self.buffer.length()
    }

    fn execute(
        &self,
        context: &<Self::Backend as Backend>::Context,
        operations: &[SparseBufferOperation],
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        Err(CpuError::NotSupported)
    }
}
