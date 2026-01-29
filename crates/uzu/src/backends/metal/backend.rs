use metal::{MTLBuffer, MTLCommandBuffer};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{MTLContext, MTLError, MetalKernels};
use crate::backends::common::Backend;

pub struct Metal;

impl Backend for Metal {
    type NativeBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type Context = MTLContext;
    type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
    type Kernels = MetalKernels;
    type Error = MTLError;
}
