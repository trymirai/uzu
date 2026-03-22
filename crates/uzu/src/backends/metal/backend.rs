use metal::{MTLBuffer, MTLEvent};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{command_buffer::MetalCommandBuffer, context::MetalContext, error::MetalError, kernel::dsl::MetalKernels};
use crate::backends::common::Backend;

#[derive(Debug, Clone)]
pub struct Metal;

impl Backend for Metal {
    type Context = MetalContext;
    type CommandBuffer = MetalCommandBuffer;
    type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type Event = Retained<ProtocolObject<dyn MTLEvent>>;
    type Kernels = MetalKernels;
    type Error = MetalError;

    const MIN_ALLOCATION_ALIGNMENT: usize = 64;
    // Metal's set_bytes supports up to 4KB per bound value.
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbytes(_:length:index:)?language=objc
    const MAX_INLINE_BYTES: usize = 4096;
}
