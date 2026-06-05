use metal::MTLBuffer;
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::{command_buffer::MetalCommandBuffer, context::MetalContext, error::MetalError, kernel::MetalKernels};
use crate::backends::{common::Backend, metal::sparse::MetalSparseBuffer};

#[derive(Debug, Clone)]
pub struct Metal;

impl Backend for Metal {
    type Context = MetalContext;
    type CommandBuffer = MetalCommandBuffer;
    type DenseBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
    type SparseBuffer = MetalSparseBuffer;
    type Kernels = MetalKernels;
    type Error = MetalError;

    const MIN_ALLOCATION_ALIGNMENT: usize = 4;
    const MAX_ALLOCATION_ALIGNMENT: usize = 64;
    const ALLOCATION_GRANULARITY: usize = 8 * 1024 * 1024;
    // Metal's set_bytes supports up to 4KB per bound value.
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setbytes(_:length:index:)?language=objc
    const MAX_INLINE_BYTES: usize = 4096;
}
