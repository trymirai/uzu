use metal::{MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLComputeCommandEncoder, MTLEvent};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{context::MetalContext, error::MetalError, kernel::dsl::MetalKernels};
use crate::backends::common::Backend;

#[derive(Debug, Clone)]
pub struct Metal;

impl Backend for Metal {
    type Context = MetalContext;
    type NativeBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
    type ComputeEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;
    type CopyEncoder = ProtocolObject<dyn MTLBlitCommandEncoder>;
    type Event = Retained<ProtocolObject<dyn MTLEvent>>;
    type Kernels = MetalKernels;
    type Error = MetalError;

    // Metal's set_bytes supports up to 4KB per bound value.
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbytes(_:length:index:)?language=objc
    const MAX_INLINE_BYTES: usize = 4096;
}
