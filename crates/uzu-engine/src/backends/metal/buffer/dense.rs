use std::{
    ptr::NonNull,
    sync::{Arc, atomic::Ordering},
};

use metal::{MTLBuffer, MTLDevice, MTLDeviceExt, MTLResidencySet, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::Mutex;

use crate::backends::{
    common::FileMapping,
    metal::{MetalContext, error::MetalError},
};

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
        Ok(Self::resident(context, buffer))
    }

    /// `length` bytes of `mapping` at `pages`, used in place; the mapping lives until Metal releases the buffer.
    pub(in crate::backends::metal) fn new_mapped(
        context: &MetalContext,
        mapping: Arc<FileMapping>,
        pages: NonNull<u8>,
        length: usize,
    ) -> Result<Self, MetalError> {
        let buffer = context
            .device
            .new_buffer_with_bytes_no_copy_with_deallocator(
                pages.cast(),
                length,
                MTLResourceOptions::STORAGE_MODE_SHARED,
                move |_, _| {
                    let _ = &mapping;
                },
            )
            .ok_or(MetalError::CannotCreateBuffer)?;
        Ok(Self::resident(context, buffer))
    }

    fn resident(
        context: &MetalContext,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    ) -> Self {
        let residency_set_locked = context.residency_set.lock();
        residency_set_locked.add_allocation(buffer.as_ref());
        residency_set_locked.commit();
        residency_set_locked.request_residency();
        drop(residency_set_locked);

        context.peak_memory_usage.fetch_max(context.device.current_allocated_size(), Ordering::Relaxed);

        Self {
            buffer,
            residency_set: context.residency_set.clone(),
        }
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
