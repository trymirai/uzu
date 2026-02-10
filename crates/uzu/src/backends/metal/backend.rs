use metal::{MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLComputeCommandEncoder};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{MTLContext, MTLError, MetalKernels};
use crate::backends::common::Backend;

#[derive(Debug, Clone)]
pub struct Metal;

impl Backend for Metal {
    type Context = MTLContext;
    type NativeBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
    type ComputeEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;
    type CopyEncoder = ProtocolObject<dyn MTLBlitCommandEncoder>;
    type Kernels = MetalKernels;
    type Error = MTLError;
}
