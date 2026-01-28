use metal::{MTLBuffer, MTLCommandBuffer};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{MTLContext, MTLError, MetalKernels};
use crate::backends::common::Backend;

pub struct Metal;

impl Backend for Metal {
    type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type ResourceOptions = MTLResourceOptions;
    type Device = Retained<ProtocolObject<dyn MTLDevice>>;
    type Context = MTLContext;
    type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
    type Kernels = MetalKernels;
    type Error = MTLError;
}
