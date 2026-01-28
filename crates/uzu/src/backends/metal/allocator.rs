use std::ops::Deref;

use metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};

use super::Metal;
use crate::backends::common::{Allocator, Buffer};

impl Buffer for Retained<ProtocolObject<dyn MTLBuffer>> {
    fn length(&self) -> usize {
        self.deref().length()
    }

    fn id(&self) -> usize {
        Retained::as_ptr(self) as usize
    }
}

pub fn new_allocator(
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    options: MTLResourceOptions,
) -> Allocator<Metal> {
    Allocator::new(device, options)
}
