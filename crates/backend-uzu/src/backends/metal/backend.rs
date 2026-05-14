use std::any::Any;

use metal::{MTLBuffer, MTLEvent};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{command_buffer::MetalCommandBuffer, context::MetalContext, error::MetalError, kernel::MetalKernels};
use crate::backends::{
    common::{Backend, Buffer},
    metal::sparse::MetalSparseBuffer,
};

#[derive(Debug, Clone)]
pub struct Metal;

impl Metal {
    pub fn buffer_downcast<B: Buffer<Backend = Self>>(buffer: &B) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        let buffer = buffer as &dyn Any;
        if let Some(buffer) = buffer.downcast_ref::<<Self as Backend>::DenseBuffer>() {
            buffer
        } else if let Some(buffer) = buffer.downcast_ref::<MetalSparseBuffer>() {
            buffer.mtl_buffer()
        } else {
            unreachable!("Unsupported Metal buffer type")
        }
    }
}

impl Backend for Metal {
    type Context = MetalContext;
    type CommandBuffer = MetalCommandBuffer;
    type DenseBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type SparseBuffer = MetalSparseBuffer;
    type Event = Retained<ProtocolObject<dyn MTLEvent>>;
    type Kernels = MetalKernels;
    type Error = MetalError;

    const MIN_ALLOCATION_ALIGNMENT: usize = 64;
    // Metal's set_bytes supports up to 4KB per bound value.
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbytes(_:length:index:)?language=objc
    const MAX_INLINE_BYTES: usize = 4096;
}
