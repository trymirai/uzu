use std::any::Any;

use metal::MTLBuffer;
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::{
    common::{Backend, Buffer},
    metal::{Metal, sparse::MetalSparseBuffer},
};

impl dyn Buffer<Backend = Metal> {
    pub fn downcast(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        let buffer = self as &dyn Any;
        if let Some(buffer) = buffer.downcast_ref::<<Metal as Backend>::DenseBuffer>() {
            buffer
        } else if let Some(buffer) = buffer.downcast_ref::<MetalSparseBuffer>() {
            buffer.mtl_buffer()
        } else {
            unreachable!("Unsupported Metal buffer type")
        }
    }
}
