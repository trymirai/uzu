use std::any::Any;

use metal::MTLBuffer;
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::{
    common::{Backend, Buffer},
    metal::{Metal, sparse::MetalSparseBuffer},
};

pub trait BufferDowncastExt: Buffer<Backend = Metal> {
    fn downcast(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>>;
}

impl<B: Buffer<Backend = Metal>> BufferDowncastExt for B {
    fn downcast(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        let buffer = self as &dyn Any;
        if let Some(buffer) = buffer.downcast_ref::<<<B as Buffer>::Backend as Backend>::DenseBuffer>() {
            buffer
        } else if let Some(buffer) = buffer.downcast_ref::<MetalSparseBuffer>() {
            buffer.mtl_buffer()
        } else {
            unreachable!("Unsupported Metal buffer type")
        }
    }
}
