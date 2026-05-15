use std::{any::Any, cell::UnsafeCell, pin::Pin};

use crate::backends::{
    common::{Backend, Buffer},
    cpu::Cpu,
};

pub trait BufferDowncastExt: Buffer<Backend = Cpu> {
    fn downcast(&self) -> &UnsafeCell<Pin<Box<[u8]>>>;
}

impl<B: Buffer<Backend = Cpu>> BufferDowncastExt for B {
    fn downcast(&self) -> &UnsafeCell<Pin<Box<[u8]>>> {
        let buffer = self as &dyn Any;
        if let Some(buffer) = buffer.downcast_ref::<<<B as Buffer>::Backend as Backend>::DenseBuffer>() {
            buffer
        } else {
            unreachable!("Unsupported Cpu buffer type")
        }
    }
}
