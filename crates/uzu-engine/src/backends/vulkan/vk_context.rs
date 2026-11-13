use std::{
    convert::Infallible,
    path::Path,
    sync::{Arc, Weak},
};

use ash::{ext, vk};
use bytemuck::AnyBitPattern;
use futures::executor::block_on;
use vk_mem::Alloc;

use crate::backends::{
    common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferPending, Context, CpuBuffer, DeviceCapabilities,
        allocator::pool::PoolAllocator,
    },
    vulkan::{
        API_VERSION_1_4, Error, VkBuffer, VkBufferCpuGpu, VkBufferError, VkBufferMemory, VkCommandBufferEncoding,
        VkComputePipeline, VkComputePipelineKey, VkContextError, VkDeviceCapabilities, VkInstance, VkPhysicalDevice,
        VkPipelines, VkSparseBuffer, VkSubmitter, Vulkan,
    },
};

const BUFFER_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS.as_raw(),
);

// Fields drop in declaration order, so everything created from the device comes before it and the instance is last.
// https://doc.rust-lang.org/reference/destructors.html
pub struct VkContext {
    weak_self: Weak<VkContext>,
    submitter: VkSubmitter,
    pipelines: VkPipelines,
    memory_allocator: vk_mem::Allocator,
    debug_labels: Option<ext::debug_utils::Device>,
    device: VkDevice,
    physical_device: VkPhysicalDevice,
    instance: VkInstance,
}

impl VkContext {
    pub fn arc(&self) -> Arc<Self> {
        self.weak_self.upgrade().expect("a VkContext is always owned by the Arc that created it")
    }

    pub fn device(&self) -> &ash::Device {
        &self.device.0
    }

    pub fn physical_device(&self) -> &VkPhysicalDevice {
        &self.physical_device
    }

    pub fn submitter(&self) -> &VkSubmitter {
        &self.submitter
    }

    pub fn debug_labels(&self) -> Option<&ext::debug_utils::Device> {
        self.debug_labels.as_ref()
    }

    pub fn take_validation_error(&self) -> Option<String> {
        self.instance.take_validation_error()
    }

    pub fn compute_pipeline(
        &self,
        key: VkComputePipelineKey,
    ) -> Result<Arc<VkComputePipeline>, Error> {
        self.pipelines.get(self, key)
    }

    pub fn allocate_buffer(
        &self,
        size: vk::DeviceSize,
        memory: VkBufferMemory,
    ) -> Result<(vk::Buffer, vk_mem::Allocation, vk_mem::AllocationInfo), Error> {
        if size == 0 {
            return Err(VkBufferError::Empty.into());
        }
        let buffer_info = vk::BufferCreateInfo::default().size(size).usage(BUFFER_USAGE);
        let discrete = self.physical_device.capabilities.contains(VkDeviceCapabilities::DISCRETE);
        let allocation_info = memory.allocation_create_info(discrete);
        let allocator = &self.memory_allocator;
        let create = || unsafe { allocator.create_buffer(&buffer_info, &allocation_info) };
        let (buffer, allocation) = match create() {
            Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY) => {
                self.wait_for_submitted()?;
                create()
            },
            created => created,
        }
        .map_err(VkBufferError::Allocation)?;
        Ok((buffer, allocation, allocator.get_allocation_info(&allocation)))
    }

    fn wait_for_submitted(&self) -> Result<(), Error> {
        let submitted = self.create_command_buffer(None, None)?.end_encoding().try_submit()?;
        block_on(submitted.wait_until_completed())?;
        Ok(())
    }

    pub unsafe fn free_buffer(
        &self,
        buffer: vk::Buffer,
        allocation: &mut vk_mem::Allocation,
    ) {
        unsafe { self.memory_allocator.destroy_buffer(buffer, allocation) };
    }
}

impl Context for VkContext {
    type Backend = Vulkan;

    async fn new() -> Result<Arc<Self>, Error> {
        let instance = VkInstance::new(cfg!(test) && !cfg!(target_os = "android"))?;
        let (physical_device, queue_family_index) = VkPhysicalDevice::select(instance.entry(), instance.handle())?;
        let device = VkDevice(physical_device.create_device(instance.handle(), queue_family_index)?);
        let debug_labels = instance.debug_labels().then(|| ext::debug_utils::Device::new(instance.handle(), &device.0));
        let mut allocator_info = vk_mem::AllocatorCreateInfo::new(instance.handle(), &device.0, physical_device.handle);
        allocator_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
        allocator_info.vulkan_api_version = API_VERSION_1_4;
        let memory_allocator =
            unsafe { vk_mem::Allocator::new(allocator_info) }.map_err(VkContextError::MemoryAllocatorCreate)?;
        let pipelines = VkPipelines::new(device.0.clone())?;
        let submitter = VkSubmitter::new(device.0.clone(), queue_family_index)?;

        Ok(Arc::new_cyclic(|weak_self| Self {
            weak_self: weak_self.clone(),
            submitter,
            pipelines,
            memory_allocator,
            debug_labels,
            device,
            physical_device,
            instance,
        }))
    }

    fn device_name(&self) -> Option<&str> {
        Some(&self.physical_device.name)
    }

    fn create_command_buffer(
        &self,
        name: Option<&str>,
        allocation_pool: Option<Arc<<Vulkan as Backend>::AllocationPool>>,
    ) -> Result<VkCommandBufferEncoding, Error> {
        let allocation_pool = allocation_pool.unwrap_or_else(|| self.create_allocation_pool());
        VkCommandBufferEncoding::new(self.arc(), allocation_pool, name)
    }

    fn create_buffer_gpu(
        &self,
        size: usize,
    ) -> Result<Arc<VkBuffer>, Error> {
        VkBuffer::new(self.arc(), size)
    }

    fn create_buffer_cpu_gpu(
        &self,
        size: usize,
    ) -> Result<VkBufferCpuGpu, Error> {
        VkBufferCpuGpu::new(self.arc(), size, VkBufferMemory::Staging)
    }

    fn create_sparse_buffer(
        &self,
        _capacity: usize,
    ) -> Result<VkSparseBuffer, Error> {
        Err(Error::NotSupported)
    }

    fn copyout<T: AnyBitPattern>(
        &self,
        buffer: impl BufferRef<Backend = Vulkan, Buffer = Arc<VkBuffer>>,
    ) -> impl Future<Output = Result<Vec<T>, Error>> + Send {
        let submitted = (buffer.size() > 0)
            .then(|| {
                let mut readback = VkBufferCpuGpu::new(self.arc(), buffer.size(), VkBufferMemory::Readback)?;
                let mut encoding = self.create_command_buffer(None, None)?;
                encoding.encode_copy(buffer, &mut readback);
                Ok::<_, Error>((encoding.end_encoding().try_submit()?, readback))
            })
            .transpose();
        async move {
            let Some((pending, readback)) = submitted? else {
                return Ok(Vec::new());
            };
            pending.wait_until_completed().await?;
            Ok(readback.copyout())
        }
    }

    fn create_allocation_pool(&self) -> Arc<<Vulkan as Backend>::AllocationPool> {
        PoolAllocator::new()
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        None
    }

    fn start_capture(
        &self,
        _output_path: &Path,
    ) -> Result<Infallible, Error> {
        Err(Error::NotSupported)
    }

    fn device_capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::empty()
    }
}

struct VkDevice(ash::Device);

impl Drop for VkDevice {
    fn drop(&mut self) {
        unsafe { self.0.destroy_device(None) };
    }
}
