use std::any::Any;

use crate::backends::{
    common::{Backend, Buffer},
    cpu::{Cpu, dense_buffer::CpuBuffer},
};

impl dyn Buffer<Backend = Cpu> {
    pub fn downcast(&self) -> &CpuBuffer {
        let buffer = self as &dyn Any;
        if let Some(buffer) = buffer.downcast_ref::<<Cpu as Backend>::DenseBuffer>() {
            buffer
        } else {
            unreachable!("Unsupported Cpu buffer type")
        }
    }
}
