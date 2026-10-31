use std::{ops::Range, os::raw::c_void, ptr::NonNull, sync::Arc};

use ash::vk;

use crate::{
    backends::{
        common::{
            Buffer, BufferRef, CpuBuffer, GlobalGpuBuffer, GpuBuffer,
            allocator::{bump::BumpAllocation, pool::PoolAllocation},
        },
        vulkan::{Error, VkBufferMemory, VkContext, Vulkan},
    },
    utils::downcast::downcast_ref,
};

/// Storage buffer in `VkBufferMemory`; host-accessed memory is permanently mapped and host-coherent.
pub struct VkBuffer {
    context: Arc<VkContext>,
    allocation: vk_mem::Allocation,
    buffer: vk::Buffer,
    device_address: vk::DeviceAddress,
    cpu_ptr: Option<NonNull<c_void>>,
    size: vk::DeviceSize,
}

// SAFETY: the mapped address is the only field that is not Send and Sync, and the buffer never dereferences it:
// `cpu_ptr` leaves synchronizing host access to its callers. VMA frees the allocation from any thread.
unsafe impl Send for VkBuffer {}
unsafe impl Sync for VkBuffer {}

impl VkBuffer {
    pub fn new(
        context: Arc<VkContext>,
        size: vk::DeviceSize,
        memory: VkBufferMemory,
    ) -> Result<Self, Error> {
        let (buffer, allocation, info) = context.allocate_buffer(size, memory)?;
        let cpu_ptr = (memory != VkBufferMemory::Device)
            .then(|| NonNull::new(info.mapped_data).expect("VMA maps host-visible MAPPED allocations"));
        let device_address = unsafe {
            context.device().get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        Ok(Self {
            context,
            allocation,
            buffer,
            device_address,
            cpu_ptr,
            size,
        })
    }

    pub fn context(&self) -> &Arc<VkContext> {
        &self.context
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    /// Host-coherent mapping of the whole buffer. Dereferencing it must not overlap GPU work that
    /// accesses the same bytes: read or write only before submission or after completion.
    ///
    /// # Panics
    /// If the buffer is in device memory, which `Context::copyout` reads.
    pub fn cpu_ptr(&self) -> NonNull<c_void> {
        self.cpu_ptr.expect("device-local Vulkan buffers are not CPU-accessible")
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// The `VkBuffer` behind any buffer of the Vulkan backend, with `range` of it in that buffer's bytes.
    pub fn resolve(
        buffer: &(impl Buffer<Backend = Vulkan> + ?Sized),
        range: std::range::Range<usize>,
    ) -> (&Arc<VkBuffer>, Range<u64>) {
        let (buffer, offset) = if let Some(buffer) = downcast_ref::<Arc<VkBuffer>>(buffer) {
            (buffer, 0)
        } else if let Some(allocation) = downcast_ref::<BumpAllocation<Arc<VkBuffer>>>(buffer) {
            (allocation.page(), allocation.range().start)
        } else if let Some(allocation) = downcast_ref::<PoolAllocation<Arc<VkBuffer>>>(buffer) {
            (allocation.page(), allocation.range().start)
        } else {
            unreachable!("unsupported Vulkan buffer type")
        };
        (buffer, (offset + range.start) as u64..(offset + range.end) as u64)
    }

    pub fn resolve_ref<'a>(buffer: impl BufferRef<Backend = Vulkan> + 'a) -> (&'a Arc<Self>, Range<u64>) {
        let (buffer, range) = buffer.parts();
        Self::resolve(buffer, range)
    }
}

impl Drop for VkBuffer {
    fn drop(&mut self) {
        // SAFETY: allocated by `new` on this context; command buffers retain the `Arc` until their GPU work completes.
        unsafe { self.context.free_buffer(self.buffer, &mut self.allocation) };
    }
}

impl Buffer for Arc<VkBuffer> {
    type Backend = Vulkan;

    fn size(&self) -> usize {
        VkBuffer::size(self) as usize
    }
}

impl GpuBuffer for Arc<VkBuffer> {}

/// Constant pages are host-visible; device-local buffers panic in `cpu_ptr`.
impl CpuBuffer for Arc<VkBuffer> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        VkBuffer::cpu_ptr(self)
    }
}

impl GlobalGpuBuffer for Arc<VkBuffer> {}
