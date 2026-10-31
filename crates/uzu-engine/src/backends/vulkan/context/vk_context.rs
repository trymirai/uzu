use std::{
    collections::HashMap,
    convert::Infallible,
    ffi::{CStr, c_char, c_void},
    path::{Path, PathBuf},
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use ash::{ext, khr, vk};
use bytemuck::AnyBitPattern;
use parking_lot::{Mutex, MutexGuard};
use vk_mem::Alloc;

use crate::backends::{
    common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferPending, Context, CpuBuffer, DeviceCapabilities,
        allocator::pool::PoolAllocator,
    },
    vulkan::{
        Error, VkBuffer, VkBufferError, VkBufferInitializer, VkBufferMemory, VkCommandBufferEncoding,
        VkCommandBufferError, VkCommandBufferResources, VkCommandBufferSubmission, VkComputePipeline,
        VkComputePipelineKey, VkContextError, VkCooperativeMatrix2Features, VkDebugMessages, VkDeviceCapabilities,
        VkKernelError, VkPhysicalDevice, VkShaderBfloat16Features, VkSparseBuffer, Vulkan,
    },
};

const VK_LAYER_KHRONOS_VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";
/// The only Vulkan version the backend targets; Ash 0.38 predates it.
const API_VERSION_1_4: u32 = vk::make_api_version(0, 1, 4, 0);
/// https://docs.vulkan.org/refpages/latest/refpages/index.html
pub struct VkContext {
    weak_self: Weak<VkContext>,
    _entry: ash::Entry,
    instance: ash::Instance,
    /// The debug messenger's user data, so it outlives the instance.
    debug_messages: Box<VkDebugMessages>,
    physical_device: VkPhysicalDevice,
    command_buffer_resources: Mutex<Vec<VkCommandBufferResources>>,
    /// Generated pipelines by recipe, held weakly: each lives while an owner or pending command buffer keeps it.
    compute_pipelines: Mutex<HashMap<VkComputePipelineKey, Weak<VkComputePipeline>>>,
    /// Compiled pipelines, loaded from the file and saved back to it with the data size it loaded.
    pipeline_cache: vk::PipelineCache,
    pipeline_cache_file: Option<(PathBuf, usize)>,
    debug_utils: ext::debug_utils::Instance,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    /// Command-buffer labels, available with validation.
    debug_utils_device: Option<ext::debug_utils::Device>,
    timestamp_valid_bits: u32,
    /// Reads the device clock, where the device has calibrated timestamps.
    calibrated_timestamps: Option<khr::calibrated_timestamps::Device>,
    device: ash::Device,
    memory_allocator: Option<vk_mem::Allocator>,
    peak_buffer_bytes: AtomicUsize,
    /// The scratch pool every allocation pool shares while one is alive. Each command buffer begins with a barrier
    /// against all earlier submissions to the queue, so pages released by one serve the next, and steady decoding
    /// allocates no device memory. Weak, as its pages hold the context.
    allocation_pool: Mutex<Weak<PoolAllocator<Arc<VkBuffer>>>>,
    /// Command buffers recording or ended but not submitted: an encoding submits segments early only while it is the
    /// only one, so they never pass a command buffer recorded or ended before them.
    unsubmitted_command_buffers: AtomicUsize,
    queue: Mutex<vk::Queue>,
    queue_family_index: u32,
    /// Submissions keep the context alive, so its completion thread is idle by the time the context drops this sender.
    completion: mpsc::Sender<VkCommandBufferSubmission>,
    /// A failed upload, where the common contract has no error to return (`GlobalBufferCpuGpu::handoff`): every later
    /// command buffer fails with it, as the uploaded buffers hold no data.
    upload_failure: Mutex<Option<String>>,
}
impl VkContext {
    /// `pipeline_cache_path` is where compiled pipelines persist between runs; `None` compiles each one again in every
    /// process.
    fn with_settings(
        with_validation: bool,
        pipeline_cache_path: Option<PathBuf>,
    ) -> Result<Arc<Self>, VkContextError> {
        let completion =
            VkCommandBufferSubmission::spawn_completion_thread().map_err(VkContextError::CompletionThreadSpawn)?;
        let debug_messages = Box::new(VkDebugMessages::default());

        let entry = unsafe { ash::Entry::load() }.map_err(VkContextError::EntryLoad)?;
        let instance = create_instance(&entry, with_validation.then_some(&*debug_messages))?;
        let debug_utils = ext::debug_utils::Instance::new(&entry, &instance);
        let debug_messenger = if with_validation {
            Some(
                unsafe { debug_utils.create_debug_utils_messenger(&debug_messenger_info(&debug_messages), None) }
                    .map_err(VkContextError::DebugMessengerCreate)
                    .inspect_err(|_| unsafe { instance.destroy_instance(None) })?,
            )
        } else {
            None
        };
        let dispose_instance = || unsafe {
            if let Some(messenger) = debug_messenger {
                debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            instance.destroy_instance(None);
        };
        let (physical_device, queue_family_index) =
            get_physical_device(&entry, &instance).inspect_err(|_| dispose_instance())?;
        let timestamp_valid_bits = physical_device.queue_families[queue_family_index as usize].timestamp_valid_bits;
        let device =
            get_logical_device(&instance, &physical_device, queue_family_index).inspect_err(|_| dispose_instance())?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let (pipeline_cache, pipeline_cache_size) = create_pipeline_cache(&device, pipeline_cache_path.as_deref())
            .inspect_err(|_| unsafe {
                device.destroy_device(None);
                dispose_instance();
            })?;
        // The allocator's buffer budgets: the OS's limit on each heap, where the device reports it.
        let memory_budget = physical_device.capabilities.contains(VkDeviceCapabilities::MEMORY_BUDGET);
        let memory_allocator = create_memory_allocator(&instance, &device, physical_device.device, memory_budget)
            .inspect_err(|_| unsafe {
                device.destroy_pipeline_cache(pipeline_cache, None);
                device.destroy_device(None);
                dispose_instance();
            })?;
        let debug_utils_device = with_validation.then(|| ext::debug_utils::Device::new(&instance, &device));
        let calibrated_timestamps = physical_device
            .capabilities
            .contains(VkDeviceCapabilities::CALIBRATED_TIMESTAMPS)
            .then(|| khr::calibrated_timestamps::Device::new(&instance, &device));

        Ok(Arc::new_cyclic(|weak_self| Self {
            weak_self: weak_self.clone(),
            _entry: entry,
            instance,
            debug_messages,
            physical_device,
            command_buffer_resources: Mutex::new(Vec::new()),
            compute_pipelines: Mutex::new(HashMap::new()),
            pipeline_cache,
            pipeline_cache_file: pipeline_cache_path.map(|path| (path, pipeline_cache_size)),
            debug_utils,
            debug_messenger,
            debug_utils_device,
            timestamp_valid_bits,
            calibrated_timestamps,
            device,
            memory_allocator: Some(memory_allocator),
            peak_buffer_bytes: AtomicUsize::new(0),
            allocation_pool: Mutex::new(Weak::new()),
            unsubmitted_command_buffers: AtomicUsize::new(0),
            queue: Mutex::new(queue),
            queue_family_index,
            completion,
            upload_failure: Mutex::new(None),
        }))
    }

    /// Fails every later `create_command_buffer` with the first upload failure.
    pub fn fail_uploads(
        &self,
        error: Error,
    ) {
        self.upload_failure.lock().get_or_insert_with(|| error.to_string());
    }

    fn create_readback_buffer(
        &self,
        size: usize,
    ) -> Result<Arc<VkBuffer>, Error> {
        Ok(Arc::new(VkBuffer::new(self.arc(), size as u64, VkBufferMemory::Readback)?))
    }

    /// The `Arc` that `with_settings` created and that owns this context; it is alive while `&self` is.
    pub fn arc(&self) -> Arc<Self> {
        self.weak_self.upgrade().expect("a VkContext is always owned by the Arc that created it")
    }

    pub fn debug_utils(&self) -> Option<&ext::debug_utils::Device> {
        self.debug_utils_device.as_ref()
    }

    /// Returns reset command resources: a cached set when one is idle, otherwise a new set.
    pub fn acquire_command_buffer_resources(&self) -> Result<VkCommandBufferResources, Error> {
        let Some(mut resources) = self.command_buffer_resources.lock().pop() else {
            return VkCommandBufferResources::new(&self.device, self.queue_family_index);
        };
        if let Err(error) = resources.reset(&self.device) {
            unsafe { resources.destroy(&self.device) };
            return Err(error);
        }
        Ok(resources)
    }

    /// # Safety
    /// The resources must come from this context and must not be pending on the GPU.
    pub unsafe fn recycle_command_buffer_resources(
        &self,
        resources: VkCommandBufferResources,
    ) {
        self.command_buffer_resources.lock().push(resources);
    }

    /// The live pipeline of this recipe or a new one; concurrent first requests may each compile, the first one wins.
    pub fn compute_pipeline(
        &self,
        key: VkComputePipelineKey,
    ) -> Result<Arc<VkComputePipeline>, Error> {
        let live = self.compute_pipelines.lock().get(&key).and_then(Weak::upgrade);
        if let Some(pipeline) = live {
            return Ok(pipeline);
        }
        let pipeline = Arc::new(VkComputePipeline::new(self.arc(), &key)?);
        // Validation errors of the module and pipeline are this creation's, not the next completed command buffer's.
        if let Some(message) = self.take_validation_error() {
            return Err(Error::Validation(message));
        }
        // A losing `pipeline` drops after the guard.
        let published = {
            let mut pipelines = self.compute_pipelines.lock();
            pipelines.retain(|_, cached| cached.strong_count() > 0);
            match pipelines.get(&key).and_then(Weak::upgrade) {
                Some(winner) => winner,
                None => {
                    pipelines.insert(key, Arc::downgrade(&pipeline));
                    pipeline.clone()
                },
            }
        };
        Ok(published)
    }

    pub fn pipeline_cache(&self) -> vk::PipelineCache {
        self.pipeline_cache
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    fn memory_allocator(&self) -> &vk_mem::Allocator {
        self.memory_allocator.as_ref().unwrap()
    }

    /// Creates and binds a VMA buffer, then raises the peak to the bytes VMA has committed in device-memory blocks.
    pub fn allocate_buffer(
        &self,
        size: vk::DeviceSize,
        memory: VkBufferMemory,
    ) -> Result<(vk::Buffer, vk_mem::Allocation, vk_mem::AllocationInfo), Error> {
        if size == 0 {
            return Err(VkBufferError::Empty.into());
        }
        let buffer_info = vk::BufferCreateInfo::default().size(size).usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        );
        let discrete = self.physical_device.properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
        let allocation_info = memory.allocation_create_info(discrete);
        let allocator = self.memory_allocator();
        let (buffer, mut allocation) =
            unsafe { allocator.create_buffer(&buffer_info, &allocation_info) }.map_err(VkBufferError::Allocation)?;
        let budgets = allocator
            .get_heap_budgets()
            .inspect_err(|_| unsafe { allocator.destroy_buffer(buffer, &mut allocation) })
            .map_err(VkBufferError::Allocation)?;
        let committed = budgets.iter().map(|budget| budget.statistics.blockBytes as usize).sum();
        self.peak_buffer_bytes.fetch_max(committed, Ordering::Relaxed);
        Ok((buffer, allocation, allocator.get_allocation_info(&allocation)))
    }

    /// # Safety
    /// `buffer` and `allocation` must come from one `allocate_buffer` call on this context, must not be freed again,
    /// and no pending GPU work or host pointer may still use them.
    pub unsafe fn free_buffer(
        &self,
        buffer: vk::Buffer,
        allocation: &mut vk_mem::Allocation,
    ) {
        unsafe { self.memory_allocator().destroy_buffer(buffer, allocation) };
    }

    /// Hold the guard only for queue submission: Vulkan requires external queue synchronization.
    pub fn queue(&self) -> MutexGuard<'_, vk::Queue> {
        self.queue.lock()
    }

    pub fn completion(&self) -> &mpsc::Sender<VkCommandBufferSubmission> {
        &self.completion
    }

    /// Incremented by each encoding and executable for its lifetime.
    pub fn unsubmitted_command_buffers(&self) -> &AtomicUsize {
        &self.unsubmitted_command_buffers
    }

    /// The time `ticks` of the device's timestamps take; tick differences wrap at the queue's valid bits.
    pub fn timestamp_duration(
        &self,
        ticks: u64,
    ) -> Duration {
        // Vulkan 1.4's timestampComputeAndGraphics gives compute queues 36 to 64 valid bits.
        let ticks = ticks & (u64::MAX >> (64 - self.timestamp_valid_bits));
        let period = f64::from(self.physical_device.properties.limits.timestamp_period);
        Duration::from_nanos((ticks as f64 * period).round() as u64)
    }

    /// One observation of the device clock: its value in timestamp ticks and the host instant halfway through reading
    /// it. Without calibrated timestamps the clock is unknown.
    pub fn device_clock(&self) -> Result<(u64, Instant), Error> {
        let calibrated = self.calibrated_timestamps.as_ref().ok_or(VkCommandBufferError::UncalibratedTimestamps)?;
        let info = [vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::DEVICE)];
        let before = Instant::now();
        let (ticks, _) =
            unsafe { calibrated.get_calibrated_timestamps(&info) }.map_err(VkCommandBufferError::DeviceClock)?;
        Ok((ticks[0], before + before.elapsed() / 2))
    }

    pub fn physical_device(&self) -> &VkPhysicalDevice {
        &self.physical_device
    }

    /// Fails unless a workgroup of `size` is within the device's limits and, with a required subgroup size, made of
    /// whole subgroups of it within maxComputeWorkgroupSubgroups.
    pub fn check_work_group(
        &self,
        kernel: &'static str,
        size: [u32; 3],
        subgroup_size: Option<u32>,
    ) -> Result<(), Error> {
        let limits = &self.physical_device.properties.limits;
        let invocations = size.into_iter().try_fold(1u32, u32::checked_mul);
        if size.contains(&0)
            || size.iter().zip(limits.max_compute_work_group_size).any(|(&size, limit)| size > limit)
            || invocations.is_none_or(|count| count > limits.max_compute_work_group_invocations)
        {
            return Err(VkKernelError::WorkGroupSize {
                size,
                limit: limits.max_compute_work_group_size,
                invocations: limits.max_compute_work_group_invocations,
            }
            .into());
        }
        let subgroups = self.physical_device.subgroup_properties.max_compute_workgroup_subgroups;
        if let Some(subgroup_size) = subgroup_size
            && (!size[0].is_multiple_of(subgroup_size)
                || invocations.is_none_or(|count| count > subgroup_size.saturating_mul(subgroups)))
        {
            return Err(VkKernelError::KernelPrecondition {
                kernel,
                condition: "a workgroup of whole subgroups of its required size, within maxComputeWorkgroupSubgroups",
            }
            .into());
        }
        Ok(())
    }

    /// The validation layer's first error since the previous call; always `None` without validation.
    pub fn take_validation_error(&self) -> Option<String> {
        self.debug_messages.take_error()
    }
}
impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            for resources in self.command_buffer_resources.get_mut().drain(..) {
                resources.destroy(&self.device);
            }
            if let Some((path, loaded_size)) = &self.pipeline_cache_file {
                save_pipeline_cache(&self.device, self.pipeline_cache, path, *loaded_size);
            }
            self.device.destroy_pipeline_cache(self.pipeline_cache, None);
            self.memory_allocator = None;
            self.device.destroy_device(None);
            if let Some(messenger) = self.debug_messenger {
                self.debug_utils.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
impl Context for VkContext {
    type Backend = Vulkan;

    async fn new() -> Result<Arc<Self>, Error> {
        // Android loads layers only from an app's own package or on debuggable builds.
        let with_validation = cfg!(test) && !cfg!(target_os = "android");
        // Tests leave the user's pipeline cache alone.
        let pipeline_cache_path =
            (!cfg!(test)).then(dirs::cache_dir).flatten().map(|dir| dir.join("uzu").join("vulkan-pipeline-cache.bin"));
        Ok(VkContext::with_settings(with_validation, pipeline_cache_path)?)
    }

    fn device_name(&self) -> Option<&str> {
        self.physical_device.properties.device_name_as_c_str().ok()?.to_str().ok()
    }

    fn create_command_buffer(
        &self,
        _name: Option<&str>,
        allocation_pool: Option<Arc<<Vulkan as Backend>::AllocationPool>>,
    ) -> Result<VkCommandBufferEncoding, Error> {
        if let Some(failure) = self.upload_failure.lock().as_ref() {
            return Err(VkContextError::UploadFailed(failure.clone()).into());
        }
        let allocation_pool = allocation_pool.unwrap_or_else(|| self.create_allocation_pool());
        VkCommandBufferEncoding::new(self.arc(), allocation_pool)
    }

    fn create_buffer_gpu(
        &self,
        size: usize,
    ) -> Result<Arc<VkBuffer>, Error> {
        Ok(Arc::new(VkBuffer::new(self.arc(), size as u64, VkBufferMemory::Device)?))
    }

    fn create_buffer_cpu_gpu(
        &self,
        size: usize,
    ) -> Result<VkBufferInitializer, Error> {
        VkBufferInitializer::new(&self.arc(), size as u64)
    }

    fn create_sparse_buffer(
        &self,
        _capacity: usize,
    ) -> Result<VkSparseBuffer, Error> {
        Err(Error::NotSupported)
    }

    /// Copies `buffer` into host-visible memory after the work submitted before it, and reads it once the copy completes.
    fn copyout<T: AnyBitPattern>(
        &self,
        buffer: impl BufferRef<Backend = Vulkan, Buffer = Arc<VkBuffer>>,
    ) -> impl Future<Output = Result<Vec<T>, Error>> + Send {
        let (source, range) = VkBuffer::resolve_ref(buffer);
        let submitted = self.create_readback_buffer((range.end - range.start) as usize).and_then(|readback| {
            let mut encoding = self.create_command_buffer(None, None)?;
            encoding.encode_copy(source, range, &readback, 0)?;
            Ok((encoding.end_encoding().try_submit()?, readback))
        });
        async move {
            let (pending, readback) = submitted?;
            pending.wait_until_completed().await?;
            Ok(readback.copyout())
        }
    }

    fn create_allocation_pool(&self) -> Arc<<Vulkan as Backend>::AllocationPool> {
        let mut shared = self.allocation_pool.lock();
        shared.upgrade().unwrap_or_else(|| {
            let pool = PoolAllocator::new();
            *shared = Arc::downgrade(&pool);
            pool
        })
    }

    /// Peak VMA-committed buffer block bytes; driver-owned memory (pipelines, command pools, descriptors) is excluded.
    fn peak_memory_usage(&self) -> Option<usize> {
        Some(self.peak_buffer_bytes.load(Ordering::Relaxed))
    }

    fn start_capture(
        &self,
        _output_path: &Path,
    ) -> Result<Infallible, Error> {
        Err(Error::NotSupported)
    }

    /// Tree shapes are tuned for few Vulkan devices, so the others speculate with a listed one.
    fn device_capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::UNTUNED_SPECULATION
    }
}
fn create_instance(
    entry: &ash::Entry,
    validation: Option<&VkDebugMessages>,
) -> Result<ash::Instance, VkContextError> {
    let mut instance_extensions: Vec<*const c_char> = Vec::new();

    let mut instance_layers: Vec<*const c_char> = Vec::new();
    let mut messenger_info = None;

    if let Some(messages) = validation {
        if !is_layer_supported(entry, VK_LAYER_KHRONOS_VALIDATION)? {
            return Err(VkContextError::ValidationNotSupported);
        }
        instance_extensions.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
        instance_layers.push(VK_LAYER_KHRONOS_VALIDATION.as_ptr());
        messenger_info = Some(debug_messenger_info(messages));
    }

    let app_info = vk::ApplicationInfo::default().api_version(API_VERSION_1_4);

    let mut instance_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);
    if let Some(next) = &mut messenger_info {
        instance_info = instance_info.push_next(next);
    }

    unsafe { entry.create_instance(&instance_info, None) }.map_err(VkContextError::InstanceCreate)
}
/// The pipeline cache `path` holds and its size; empty without a file, and the driver ignores another device's data.
fn create_pipeline_cache(
    device: &ash::Device,
    path: Option<&Path>,
) -> Result<(vk::PipelineCache, usize), VkContextError> {
    let data = match path.map(|path| (path, std::fs::read(path))) {
        Some((_, Ok(data))) => data,
        Some((path, Err(error))) if error.kind() != std::io::ErrorKind::NotFound => {
            eprintln!("[Warning]: ignoring the unreadable pipeline cache {}: {error}", path.display());
            Vec::new()
        },
        _ => Vec::new(),
    };
    let info = vk::PipelineCacheCreateInfo::default().initial_data(&data);
    let cache = unsafe { device.create_pipeline_cache(&info, None) }.map_err(VkContextError::PipelineCacheCreate)?;
    Ok((cache, data.len()))
}
/// Replaces `path` through a temporary file, so no reader sees a partial cache; a cache of the loaded size is unchanged.
fn save_pipeline_cache(
    device: &ash::Device,
    cache: vk::PipelineCache,
    path: &Path,
    loaded_size: usize,
) {
    static SAVES: AtomicUsize = AtomicUsize::new(0);
    let saved = unsafe { device.get_pipeline_cache_data(cache) }.map_err(std::io::Error::other).and_then(|data| {
        if data.len() == loaded_size {
            return Ok(());
        }
        let save = SAVES.fetch_add(1, Ordering::Relaxed);
        let temporary = path.with_extension(format!("{}-{save}.tmp", std::process::id()));
        std::fs::create_dir_all(path.parent().unwrap_or(Path::new("")))?;
        std::fs::write(&temporary, &data).and_then(|()| std::fs::rename(&temporary, path)).inspect_err(|_| {
            let _ = std::fs::remove_file(&temporary);
        })
    });
    if let Err(error) = saved {
        eprintln!("[Warning]: pipeline cache {} not saved: {error}", path.display());
    }
}
fn create_memory_allocator(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: vk::PhysicalDevice,
    memory_budget: bool,
) -> Result<vk_mem::Allocator, VkContextError> {
    let mut info = vk_mem::AllocatorCreateInfo::new(instance, device, physical_device);
    info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;
    if memory_budget {
        info.flags |= vk_mem::AllocatorCreateFlags::EXT_MEMORY_BUDGET;
    }
    info.vulkan_api_version = vk::API_VERSION_1_3;
    unsafe { vk_mem::Allocator::new(info) }.map_err(VkContextError::MemoryAllocatorCreate)
}
fn get_physical_device(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> Result<(VkPhysicalDevice, u32), VkContextError> {
    let devices = unsafe { instance.enumerate_physical_devices() }.map_err(VkContextError::DeviceEnumeration)?;

    let device_opt = devices
        .into_iter()
        // Older devices must not see the Vulkan 1.4 structures that querying chains.
        .filter(|&device| unsafe { instance.get_physical_device_properties(device) }.api_version >= API_VERSION_1_4)
        .map(|device| VkPhysicalDevice::new(entry, instance, device))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .filter_map(|physical_device| physical_device.compute_queue_family().map(|queue| (physical_device, queue)))
        .min_by_key(|(physical_device, _)| match physical_device.properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            vk::PhysicalDeviceType::CPU => 3,
            vk::PhysicalDeviceType::OTHER => 4,
            _ => 5,
        });
    device_opt.ok_or(VkContextError::NoSuitableDevice)
}
fn get_logical_device(
    instance: &ash::Instance,
    physical_device: &VkPhysicalDevice,
    queue_family_index: u32,
) -> Result<ash::Device, VkContextError> {
    let queue_priorities = [1.0_f32];
    let queue_create_info =
        vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family_index).queue_priorities(&queue_priorities);

    let capabilities = physical_device.capabilities;
    let bf16_type = capabilities.contains(VkDeviceCapabilities::BF16_TYPE);
    let matrices =
        capabilities.intersects(VkDeviceCapabilities::BF16_TENSOR_CORES | VkDeviceCapabilities::INT8_TENSOR_CORES);
    let element_fold = capabilities.contains(VkDeviceCapabilities::INT8_ELEMENT_FOLD);
    let mut device_extensions = Vec::new();
    if capabilities.contains(VkDeviceCapabilities::MEMORY_BUDGET) {
        device_extensions.push(ext::memory_budget::NAME.as_ptr());
    }
    if matrices {
        device_extensions.push(khr::cooperative_matrix::NAME.as_ptr());
    }
    if bf16_type {
        device_extensions.push(VkShaderBfloat16Features::EXTENSION_NAME.as_ptr());
    }
    if element_fold {
        device_extensions.push(VkCooperativeMatrix2Features::EXTENSION_NAME.as_ptr());
    }
    if capabilities.contains(VkDeviceCapabilities::CALIBRATED_TIMESTAMPS) {
        device_extensions.push(khr::calibrated_timestamps::NAME.as_ptr());
    }

    // Selection checked the optional features; Vulkan 1.4 makes the others mandatory.
    let vk10_features = vk::PhysicalDeviceFeatures::default().shader_int16(true);
    let mut vk11_features =
        vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true).storage_push_constant16(true);
    let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default()
        .shader_int8(true)
        .storage_buffer8_bit_access(true)
        .storage_push_constant8(true)
        .shader_float16(true)
        .shader_subgroup_extended_types(true)
        .buffer_device_address(true)
        .host_query_reset(true)
        .vulkan_memory_model(true)
        .vulkan_memory_model_device_scope(true);
    let mut vk13_features = vk::PhysicalDeviceVulkan13Features::default()
        .maintenance4(true)
        .synchronization2(true)
        .subgroup_size_control(true)
        .compute_full_subgroups(true)
        .shader_integer_dot_product(true);

    let mut matrix_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
    let mut bfloat16_features = VkShaderBfloat16Features::default();
    bfloat16_features.shader_bfloat16_type = vk::TRUE;
    bfloat16_features.shader_bfloat16_cooperative_matrix =
        capabilities.contains(VkDeviceCapabilities::BF16_TENSOR_CORES).into();
    let mut matrix2_features = VkCooperativeMatrix2Features::default();
    matrix2_features.cooperative_matrix_per_element_operations = vk::TRUE;
    let mut device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_create_info))
        .enabled_extension_names(&device_extensions)
        .enabled_features(&vk10_features)
        .push_next(&mut vk11_features)
        .push_next(&mut vk12_features)
        .push_next(&mut vk13_features);
    if matrices {
        device_create_info = device_create_info.push_next(&mut matrix_features);
    }
    if bf16_type {
        device_create_info = device_create_info.push_next(&mut bfloat16_features);
    }
    if element_fold {
        device_create_info = device_create_info.push_next(&mut matrix2_features);
    }
    unsafe { instance.create_device(physical_device.device, &device_create_info, None) }
        .map_err(VkContextError::DeviceCreate)
}
fn is_layer_supported(
    entry: &ash::Entry,
    layer: &CStr,
) -> Result<bool, VkContextError> {
    let layer_properties =
        unsafe { entry.enumerate_instance_layer_properties() }.map_err(VkContextError::LayerEnumeration)?;
    Ok(layer_properties.iter().any(|properties| properties.layer_name_as_c_str().is_ok_and(|name| name == layer)))
}

fn debug_messenger_info(messages: &VkDebugMessages) -> vk::DebugUtilsMessengerCreateInfoEXT<'_> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(debug_message_callback))
        .user_data(messages as *const VkDebugMessages as *mut c_void)
}

unsafe extern "system" fn debug_message_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    p_user_data: *mut c_void,
) -> vk::Bool32 {
    let messages = unsafe { &*(p_user_data as *const VkDebugMessages) };
    messages.receive(message_severity, message_type, unsafe { CStr::from_ptr((*p_callback_data).p_message) });
    vk::FALSE
}
