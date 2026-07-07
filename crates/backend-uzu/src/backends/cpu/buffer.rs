use std::{any::Any, cell::UnsafeCell, pin::Pin};

use crate::backends::{
    common::{Backend, Buffer},
    cpu::Cpu,
};

impl dyn Buffer<Backend = Cpu> {
    pub fn downcast(&self) -> &UnsafeCell<Pin<Box<[u8]>>> {
        let buffer = self as &dyn Any;
        if let Some(buffer) = buffer.downcast_ref::<<Cpu as Backend>::DenseBuffer>() {
            buffer
        } else {
            unreachable!("Unsupported Cpu buffer type")
        }
    }
}
