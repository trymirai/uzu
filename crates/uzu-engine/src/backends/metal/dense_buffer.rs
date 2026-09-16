use std::{os::raw::c_void, ptr::NonNull, sync::Arc};

use metal::{MTLBuffer, MTLResidencySet};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::Mutex;

use crate::backends::{
    common::{Buffer, DenseBuffer},
    metal::Metal,
};

#[derive(Debug)]
pub struct MetalDenseBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    residency_set: Arc<Mutex<Retained<ProtocolObject<dyn MTLResidencySet>>>>,
}

impl MetalDenseBuffer {
    pub(super) fn new(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        residency_set: Arc<Mutex<Retained<ProtocolObject<dyn MTLResidencySet>>>>,
    ) -> Self {
        let residency_set_locked = residency_set.lock();
        residency_set_locked.add_allocation(buffer.as_ref());
        residency_set_locked.commit();
        residency_set_locked.request_residency();
        drop(residency_set_locked);

        Self {
            buffer,
            residency_set,
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

impl Buffer for MetalDenseBuffer {
    type Backend = Metal;

    fn gpu_ptr(&self) -> usize {
        self.buffer.gpu_address() as usize
    }

    fn size(&self) -> usize {
        self.buffer.length()
    }
}

impl DenseBuffer for MetalDenseBuffer {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        self.buffer.contents()
    }
}
