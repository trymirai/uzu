use std::{os::raw::c_void, ptr::NonNull, sync::Arc};

use metal::{MTLBuffer, MTLResidencySet};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::Mutex;

use crate::backends::{
    common::{Buffer, CpuBuffer, GpuBuffer},
    metal::{Metal, context::MetalContext},
};

/// A Metal buffer over memory owned elsewhere (e.g. an IOSurface shared with the Neural Engine).
///
/// It is resident for as long as it lives and is never recycled by uzu's allocators; whoever creates
/// it must keep the underlying memory alive at least as long.
pub struct MetalExternalBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    residency_set: Arc<Mutex<Retained<ProtocolObject<dyn MTLResidencySet>>>>,
}

impl MetalExternalBuffer {
    pub fn new(
        context: &MetalContext,
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    ) -> Self {
        let residency_set_locked = context.residency_set.lock();
        residency_set_locked.add_allocation(buffer.as_ref());
        residency_set_locked.commit();
        residency_set_locked.request_residency();
        drop(residency_set_locked);
        Self {
            buffer,
            residency_set: context.residency_set.clone(),
        }
    }

    pub fn mtl_buffer(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.buffer
    }
}

impl Drop for MetalExternalBuffer {
    fn drop(&mut self) {
        let residency_set_locked = self.residency_set.lock();
        residency_set_locked.remove_allocation(self.buffer.as_ref());
        residency_set_locked.commit();
    }
}

impl Buffer for MetalExternalBuffer {
    type Backend = Metal;

    fn size(&self) -> usize {
        self.buffer.length()
    }
}

impl GpuBuffer for MetalExternalBuffer {}

impl CpuBuffer for MetalExternalBuffer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.buffer.contents()
    }
}
