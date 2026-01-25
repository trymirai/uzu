use metal::MTLCommandBuffer;
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::common::Backend;

use super::{MTLContext, MTLError, MetalKernels};

pub struct Metal;

impl Backend for Metal {
    type Context = MTLContext;
    type CommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
    type Kernels = MetalKernels;

    type Error = MTLError;
}
