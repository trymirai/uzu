use std::{
    convert::Infallible,
    path::Path,
    sync::{Arc, Weak},
    thread,
};

use bytemuck::AnyBitPattern;
use futures::executor::block_on;

use crate::backends::{
    common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferPending, Context, CpuBuffer, DeviceCapabilities,
        allocator::pool::PoolAllocator,
    },
    vulkan::{
        Error, VkBuffer, VkBufferCpuGpu, VkCommandBufferEncoding, VkCommandQueue, VkDevice, VkPipelines,
        VkSparseBuffer, Vulkan,
    },
};

pub struct VkContext {
    weak_self: Weak<VkContext>,
    device: Arc<VkDevice>,
    command_queue: VkCommandQueue,
    pipelines: VkPipelines,
}

impl VkContext {
    pub fn arc(&self) -> Arc<Self> {
        self.weak_self.upgrade().expect("a VkContext is always owned by the Arc that created it")
    }

    pub fn device(&self) -> &Arc<VkDevice> {
        &self.device
    }

    pub fn command_queue(&self) -> &VkCommandQueue {
        &self.command_queue
    }

    pub fn pipelines(&self) -> &VkPipelines {
        &self.pipelines
    }

    pub fn wait_for_submitted(&self) -> Result<(), Error> {
        let submitted = self.create_command_buffer(None, None)?.end_encoding().try_submit()?;
        Self::wait(submitted.wait_until_completed())?;
        Ok(())
    }

    // Blocks on a future from another thread, because `block_on` panics when the caller already runs inside one.
    pub fn wait<T: Send>(future: impl Future<Output = T> + Send) -> T {
        thread::scope(|scope| scope.spawn(|| block_on(future)).join().expect("a Vulkan wait panicked"))
    }
}

impl Context for VkContext {
    type Backend = Vulkan;

    async fn new() -> Result<Arc<Self>, Error> {
        let device = Arc::new(VkDevice::new(cfg!(test) && !cfg!(target_os = "android"))?);
        // Tests compile their pipelines without the user's cache file.
        let pipeline_cache_path = (!cfg!(test)).then(dirs::home_dir).flatten().map(|home| {
            home.join(".cache").join("uzu").join(format!("vulkan-pipeline-cache-{}.bin", env!("UZU_VULKAN_SPIRV_HASH")))
        });
        let pipelines = VkPipelines::new(device.clone(), pipeline_cache_path)?;
        let command_queue = VkCommandQueue::new(device.clone())?;
        Ok(Arc::new_cyclic(|weak_self| Self {
            weak_self: weak_self.clone(),
            device,
            command_queue,
            pipelines,
        }))
    }

    fn device_name(&self) -> Option<&str> {
        Some(&self.device.physical_device.name)
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
        VkBuffer::new(self, size)
    }

    fn create_buffer_cpu_gpu(
        &self,
        size: usize,
    ) -> Result<VkBufferCpuGpu, Error> {
        VkBufferCpuGpu::staging(self.arc(), size)
    }

    fn create_sparse_buffer(
        &self,
        _capacity: usize,
    ) -> Result<VkSparseBuffer, Error> {
        Err(Error::NotSupported)
    }

    async fn copyout<T: AnyBitPattern>(
        &self,
        buffer: impl BufferRef<Backend = Vulkan, Buffer = Arc<VkBuffer>>,
    ) -> Result<Vec<T>, Error> {
        if buffer.size() == 0 {
            return Ok(Vec::new());
        }
        let mut readback = VkBufferCpuGpu::readback(self.arc(), buffer.size())?;
        let mut encoding = self.create_command_buffer(None, None)?;
        encoding.encode_copy(buffer, &mut readback);
        encoding.end_encoding().try_submit()?.wait_until_completed().await?;
        Ok(readback.copyout())
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
