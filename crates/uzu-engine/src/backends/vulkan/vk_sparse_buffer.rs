use std::sync::Arc;

use super::{Error, VkBuffer, VkContext, Vulkan};
use crate::backends::common::{Buffer, SparseBuffer};

/// A buffer of fixed capacity whose device address never changes as `map` grows its length. Vulkan sparse residency
/// is an optional device feature the backend does not require, so the whole capacity is allocated up front: mapping
/// only checks the length against the capacity.
pub struct VkSparseBuffer {
    buffer: Arc<VkBuffer>,
}

impl VkSparseBuffer {
    pub fn new(
        context: Arc<VkContext>,
        capacity: usize,
    ) -> Result<Self, Error> {
        Ok(Self {
            buffer: Arc::new(VkBuffer::new(context, capacity as u64)?),
        })
    }

    pub fn buffer(&self) -> &Arc<VkBuffer> {
        &self.buffer
    }
}

impl Buffer for VkSparseBuffer {
    type Backend = Vulkan;

    fn size(&self) -> usize {
        VkBuffer::size(&self.buffer) as usize
    }
}

impl SparseBuffer for VkSparseBuffer {
    fn map(
        &mut self,
        until: usize,
    ) -> Result<(), Error> {
        let capacity = Buffer::size(self);
        if until > capacity {
            return Err(Error::SparseCapacity {
                requested: until,
                capacity,
            });
        }
        Ok(())
    }
}
