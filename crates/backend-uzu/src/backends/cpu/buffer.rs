use std::{any::Any, cell::UnsafeCell, pin::Pin};

use crate::backends::{
    common::{Backend, Buffer},
    cpu::Cpu,
};

pub fn cpu_buffer<B: Buffer<Backend = Cpu>>(buffer: &B) -> &UnsafeCell<Pin<Box<[u8]>>> {
    let buffer = buffer as &dyn Any;
    if let Some(buf) = buffer.downcast_ref::<<Cpu as Backend>::DenseBuffer>() {
        buf
    } else {
        unreachable!("Unsupported Cpu buffer type")
    }
}
