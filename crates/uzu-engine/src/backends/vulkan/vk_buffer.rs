use std::{ops::Range, os::raw::c_void, ptr::NonNull, sync::Arc};

use ash::vk;

use super::{Error, VkBufferError, VkContext, Vulkan};
use crate::{
    backends::common::{
        Buffer, BufferCpuAccessible, BufferRef, ConstantBuffer, GlobalBuffer, ScratchBuffer,
        allocator::{bump::BumpAllocation, pool::PoolAllocation},
    },
    utils::downcast::downcast_ref,
};

/// Storage buffer that is permanently mapped into host-visible, host-coherent memory.
pub struct VkBuffer {
    context: Arc<VkContext>,
    allocation: vk_mem::Allocation,
    buffer: vk::Buffer,
    device_address: vk::DeviceAddress,
    cpu_ptr: NonNull<c_void>,
    size: vk::DeviceSize,
}

// The mapping is owned by this buffer and stays valid until drop; host access goes through `&`/`&mut`.
unsafe impl Send for VkBuffer {}
unsafe impl Sync for VkBuffer {}

impl VkBuffer {
    pub fn new(
        context: Arc<VkContext>,
        size: vk::DeviceSize,
    ) -> Result<Self, Error> {
        let (buffer, allocation, info) = context.allocate_buffer(size)?;
        let cpu_ptr = NonNull::new(info.mapped_data).expect("VMA maps host-visible MAPPED allocations");
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

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    /// Host-coherent mapping of the whole buffer. Dereferencing it must not overlap GPU work that
    /// accesses the same bytes: read or write only before submission or after completion.
    pub fn cpu_ptr(&self) -> NonNull<c_void> {
        self.cpu_ptr
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    pub fn fill(
        &mut self,
        data: &[u8],
    ) -> Result<(), VkBufferError> {
        if data.len() as u64 > self.size {
            return Err(VkBufferError::SizeOutOfBounds {
                requested: data.len(),
                size: self.size,
            });
        }
        unsafe { std::slice::from_raw_parts_mut(self.cpu_ptr.as_ptr().cast::<u8>(), data.len()) }.copy_from_slice(data);
        Ok(())
    }

    /// # Safety
    /// No GPU work or host writer may access the buffer during the copy: every command buffer that
    /// writes it must have completed, and no other host thread may write through `cpu_ptr`.
    pub unsafe fn get_bytes(&self) -> Box<[u8]> {
        Box::from(unsafe { std::slice::from_raw_parts(self.cpu_ptr.as_ptr().cast::<u8>(), self.size as usize) })
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

impl BufferCpuAccessible for Arc<VkBuffer> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        VkBuffer::cpu_ptr(self)
    }
}

impl GlobalBuffer for Arc<VkBuffer> {}

impl Buffer for BumpAllocation<Arc<VkBuffer>> {
    type Backend = Vulkan;

    fn size(&self) -> usize {
        self.range().iter().len()
    }
}

impl BufferCpuAccessible for BumpAllocation<Arc<VkBuffer>> {
    fn cpu_ptr(&self) -> NonNull<c_void> {
        unsafe { VkBuffer::cpu_ptr(self.page()).byte_add(self.range().start) }
    }
}

impl ConstantBuffer for BumpAllocation<Arc<VkBuffer>> {}

impl Buffer for PoolAllocation<Arc<VkBuffer>> {
    type Backend = Vulkan;

    fn size(&self) -> usize {
        self.range().iter().len()
    }
}

impl ScratchBuffer for PoolAllocation<Arc<VkBuffer>> {}
