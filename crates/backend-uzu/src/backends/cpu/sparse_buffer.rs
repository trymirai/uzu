use std::{cell::UnsafeCell, pin::Pin};

use crate::backends::{
    common::{Backend, SparseBuffer},
    cpu::Cpu,
};

#[derive(Debug)]
pub struct CpuSparseBuffer {
    buffer: UnsafeCell<Pin<Box<[u8]>>>,
    capacity: usize,
    length: usize,
}

impl CpuSparseBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: UnsafeCell::new(Pin::new(Vec::<u8>::new().into_boxed_slice())),
            capacity,
            length: 0,
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

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn extend(
        &mut self,
        add_length: usize,
    ) {
        let new_length = self.length + add_length;
        let pin = self.buffer.get_mut();
        let placeholder = Pin::new(Vec::<u8>::new().into_boxed_slice());
        let old = std::mem::replace(pin, placeholder);
        let mut vec: Vec<u8> = Pin::into_inner(old).into_vec();
        vec.resize(new_length, 0);
        *pin = Pin::new(vec.into_boxed_slice());
        self.length = new_length;
    }

    fn length(&self) -> usize {
        self.length
    }
}
