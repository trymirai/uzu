use metal::{MTLBuffer, MTLCommandBuffer, MTLComputeCommandEncoder};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{MTLContext, MTLError, MetalKernels};
use crate::backends::common::Backend;

pub struct Metal;

impl Backend for Metal {
    type NativeBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type Context = MTLContext;
    type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
    type BufferRef = ProtocolObject<dyn MTLBuffer>;
    type EncoderRef = ProtocolObject<dyn MTLComputeCommandEncoder>;
    type Kernels = MetalKernels;
    type Error = MTLError;
}
