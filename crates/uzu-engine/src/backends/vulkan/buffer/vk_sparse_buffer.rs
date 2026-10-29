use std::convert::Infallible;

use crate::backends::{
    common::{Buffer, GpuBuffer, SparseBuffer},
    vulkan::{Error, Vulkan},
};

pub struct VkSparseBuffer {
    never: Infallible,
}

impl Buffer for VkSparseBuffer {
    type Backend = Vulkan;

    fn size(&self) -> usize {
        match self.never {}
    }
}

impl GpuBuffer for VkSparseBuffer {}

impl SparseBuffer for VkSparseBuffer {
    fn map(
        &mut self,
        _until: usize,
    ) -> Result<(), Error> {
        match self.never {}
    }
}
