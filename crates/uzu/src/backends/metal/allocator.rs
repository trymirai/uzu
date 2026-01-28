use std::ops::Deref;

use metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};

use crate::backends::common::{Allocator, Buffer};

use super::Metal;

impl Buffer for Retained<ProtocolObject<dyn MTLBuffer>> {
    fn length(&self) -> usize {
        self.deref().length()
    }

    fn id(&self) -> usize {
        Retained::as_ptr(self) as usize
    }
}

pub type MetalAllocator = Allocator<Metal>;

pub fn new_allocator(
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    options: MTLResourceOptions,
) -> MetalAllocator {
    Allocator::new(device, options)
}
