use std::{os::raw::c_void, ptr::NonNull, sync::Arc};

use metal::{MTLBuffer, MTLDevice, MTLDeviceExt, MTLResidencySet, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};
use parking_lot::Mutex;

use crate::backends::{
    common::{
        BufferCpuAccessible, ConstantBuffer, GlobalBuffer, ScratchBuffer,
        allocator::{Allocation, Storage},
    },
    metal::{Metal, error::MetalError},
};

#[derive(Debug)]
pub struct MetalDenseBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    residency_set: Arc<Mutex<Retained<ProtocolObject<dyn MTLResidencySet>>>>,
}

impl MetalDenseBuffer {
    pub(in crate::backends::metal) fn new(
        size: usize,
        device: &ProtocolObject<dyn MTLDevice>,
        residency_set: Arc<Mutex<Retained<ProtocolObject<dyn MTLResidencySet>>>>,
    ) -> Result<Self, MetalError> {
        let buffer =
            device.new_buffer(size, MTLResourceOptions::STORAGE_MODE_SHARED).ok_or(MetalError::CannotCreateBuffer)?;
        let residency_set_locked = residency_set.lock();
        residency_set_locked.add_allocation(buffer.as_ref());
        residency_set_locked.commit();
        residency_set_locked.request_residency();
        drop(residency_set_locked);

        Ok(Self {
            buffer,
            residency_set,
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

impl Storage for MetalDenseBuffer {
    type Backend = Metal;

    const MIN_ALLOCATION_ALIGNMENT: usize = 4;
    const MAX_ALLOCATION_ALIGNMENT: usize = 64;
    const ALLOCATION_GRANULARITY: usize = 8 * 1024 * 1024;
}

impl BufferCpuAccessible for Allocation<MetalDenseBuffer> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { self.buffer().buffer.contents().byte_add(self.offset()) }
    }
}

impl GlobalBuffer for Allocation<MetalDenseBuffer> {}
impl ConstantBuffer for Allocation<MetalDenseBuffer> {}
impl ScratchBuffer for Allocation<MetalDenseBuffer> {}
