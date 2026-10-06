use std::sync::{Arc, atomic::Ordering};

use metal::{MTLBuffer, MTLDevice, MTLDeviceExt, MTLResidencySet, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::Mutex;

use crate::backends::metal::{MetalContext, error::MetalError};

pub struct MetalDenseBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    residency_set: Arc<Mutex<Retained<ProtocolObject<dyn MTLResidencySet>>>>,
}

impl MetalDenseBuffer {
    pub(in crate::backends::metal) fn new(
        context: &MetalContext,
        size: usize,
    ) -> Result<Self, MetalError> {
        let buffer = context
            .device
            .new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or(MetalError::CannotCreateBuffer)?;

        let residency_set_locked = context.residency_set.lock();
        residency_set_locked.add_allocation(buffer.as_ref());
        residency_set_locked.commit();
        residency_set_locked.request_residency();
        drop(residency_set_locked);

        context.peak_memory_usage.fetch_max(context.device.current_allocated_size(), Ordering::Relaxed);

        Ok(Self {
            buffer,
            residency_set: context.residency_set.clone(),
        })
    }

    pub(super) fn mtl_buffer(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.buffer
    }
}

impl Drop for MetalDenseBuffer {
    fn drop(&mut self) {
        let residency_set_locked = self.residency_set.lock();
        residency_set_locked.remove_allocation(self.buffer.as_ref());
        residency_set_locked.commit();
    }
}
