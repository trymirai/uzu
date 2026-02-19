use metal::{MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLComputeCommandEncoder, MTLEvent};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{MetalContext, MetalError, MetalKernels};
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
}
