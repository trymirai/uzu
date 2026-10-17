use std::{
    collections::HashMap,
    convert::Infallible,
    ffi::{CStr, c_char, c_void},
    path::Path,
    sync::{
        Arc, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use ash::{ext, khr, vk};
use parking_lot::{Mutex, MutexGuard};
use vk_mem::Alloc;

use super::{
    Error, VkBuffer, VkBufferError, VkCommandBufferEncoding, VkCommandBufferResources, VkComputePipeline,
    VkComputePipelineKey, VkContextCreateInfo, VkContextError, VkCooperativeMatrix2Features, VkDeviceCapabilities,
    VkLogger, VkPhysicalDevice, VkPhysicalDeviceFeatures, VkShader, VkShaderBfloat16Features, VkSparseBuffer, Vulkan,
};
use crate::backends::common::{Backend, Context, DeviceCapabilities, allocator::pool::PoolAllocator};

const VK_LAYER_KHRONOS_VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";
const HOST_COHERENT_MEMORY: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw() | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
);
/// https://docs.vulkan.org/refpages/latest/refpages/index.html
pub struct VkContext {
    weak_self: Weak<VkContext>,
    _entry: ash::Entry,
    instance: ash::Instance,
    create_info: Box<VkContextCreateInfo>,
    physical_device: VkPhysicalDevice,
    command_buffer_resources: Mutex<Vec<VkCommandBufferResources>>,
    /// Generated pipelines by recipe, held weakly: each lives while an owner or pending command buffer keeps it.
    compute_pipelines: Mutex<HashMap<VkComputePipelineKey, Weak<VkComputePipeline>>>,
    debug_utils: ext::debug_utils::Instance,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    /// Command-buffer labels, available with validation.
    debug_utils_device: Option<ext::debug_utils::Device>,
    timestamp_valid_bits: u32,
    device: ash::Device,
    memory_allocator: Option<vk_mem::Allocator>,
    peak_buffer_bytes: AtomicUsize,
    queue: Mutex<vk::Queue>,
    queue_family_index: u32,
}
impl VkContext {
    pub fn with_create_info(create_info: VkContextCreateInfo) -> Result<Arc<Self>, VkContextError> {
        let create_info = Box::new(create_info);
        let api_version = vk::API_VERSION_1_3;
        let required_extensions = [
            khr::shader_float_controls::NAME,
            khr::shader_float16_int8::NAME,
            khr::shader_subgroup_extended_types::NAME,
        ];
        let required_features = VkPhysicalDeviceFeatures::all();

        let entry = unsafe { ash::Entry::load() }.map_err(VkContextError::EntryLoadingError)?;
        let instance = create_instance(&entry, api_version, &create_info)?;
        let debug_utils = ext::debug_utils::Instance::new(&entry, &instance);
        let debug_messenger = if create_info.with_validation {
            Some(
                unsafe { debug_utils.create_debug_utils_messenger(&debug_messenger_info(&create_info), None) }
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
        let physical_device = get_physical_device(&entry, &instance, &required_extensions, required_features)
            .inspect_err(|_| dispose_instance())?;
        let (device, queue_family_index, timestamp_valid_bits) =
            get_logical_device(&instance, &physical_device, &required_extensions, required_features)
                .inspect_err(|_| dispose_instance())?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        // The allocator's buffer budgets: the OS's limit on each heap, where the device reports it.
        let memory_budget = physical_device.capabilities.contains(VkDeviceCapabilities::MEMORY_BUDGET);
        let memory_allocator = create_memory_allocator(&instance, &device, physical_device.device, memory_budget)
            .inspect_err(|_| unsafe {
                device.destroy_device(None);
                dispose_instance();
            })?;
        let debug_utils_device = create_info.with_validation.then(|| ext::debug_utils::Device::new(&instance, &device));

        Ok(Arc::new_cyclic(|weak_self| Self {
            weak_self: weak_self.clone(),
            _entry: entry,
            instance,
            create_info,
            physical_device,
            command_buffer_resources: Mutex::new(Vec::new()),
            compute_pipelines: Mutex::new(HashMap::new()),
            debug_utils,
            debug_messenger,
            debug_utils_device,
            timestamp_valid_bits,
            device,
            memory_allocator: Some(memory_allocator),
            peak_buffer_bytes: AtomicUsize::new(0),
            queue: Mutex::new(queue),
            queue_family_index,
        }))
    }

    /// The `Arc` that `new` created and that owns this context; it is alive while `&self` is.
    pub fn arc(&self) -> Arc<Self> {
        self.weak_self.upgrade().expect("a VkContext is always owned by the Arc that created it")
    }

    pub fn debug_utils(&self) -> Option<&ext::debug_utils::Device> {
        self.debug_utils_device.as_ref()
    }

    /// Returns reset command resources: a cached set when one is idle, otherwise a new set.
    pub fn acquire_command_buffer_resources(&self) -> Result<VkCommandBufferResources, Error> {
        let Some(resources) = self.command_buffer_resources.lock().pop() else {
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
        spirv: &'static [u8],
        entry_point: &'static str,
        specialization_entries: &[vk::SpecializationMapEntry],
        specialization_data: &[u8],
        push_constant_size: u32,
        required_subgroup_size: Option<u32>,
    ) -> Result<Arc<VkComputePipeline>, Error> {
        let key = VkComputePipelineKey::new(
            spirv,
            entry_point,
            specialization_entries,
            specialization_data,
            push_constant_size,
            required_subgroup_size,
        );
        let live = self.compute_pipelines.lock().get(&key).and_then(Weak::upgrade);
        if let Some(pipeline) = live {
            return Ok(pipeline);
        }
        let shader = VkShader::new(self.arc(), spirv)?;
        let specialization =
            vk::SpecializationInfo::default().map_entries(specialization_entries).data(specialization_data);
        let pipeline = Arc::new(VkComputePipeline::new(
            self.arc(),
            shader.module(),
            push_constant_size,
            entry_point,
            &specialization,
            required_subgroup_size,
        )?);
        // A losing `pipeline` and the `shader` drop after the guard.
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
    ) -> Result<(vk::Buffer, vk_mem::Allocation, vk_mem::AllocationInfo), Error> {
        if size == 0 {
            return Err(Error::EmptyBuffer);
        }
        let buffer_info = vk::BufferCreateInfo::default().size(size).usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        );
        // Every buffer holds GPU data. On a discrete GPU it must be device-local, mapped through the resizable BAR, and
        // within the heap's budget, so exhausting video memory fails rather than spilling to system memory.
        let discrete = self.physical_device.properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
        let (local, budget) = if discrete {
            (vk::MemoryPropertyFlags::DEVICE_LOCAL, vk_mem::AllocationCreateFlags::WITHIN_BUDGET)
        } else {
            (vk::MemoryPropertyFlags::empty(), vk_mem::AllocationCreateFlags::empty())
        };
        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            // Sequential write, unlike random access, does not prefer cached host memory; host reads remain valid.
            flags: vk_mem::AllocationCreateFlags::MAPPED
                | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | budget,
            required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT | local,
            ..Default::default()
        };
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

    pub fn timestamp_valid_bits(&self) -> u32 {
        self.timestamp_valid_bits
    }

    pub fn physical_device(&self) -> &VkPhysicalDevice {
        &self.physical_device
    }

    pub fn logger(&self) -> &dyn VkLogger {
        self.create_info.logger.as_ref()
    }
}
impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            for resources in self.command_buffer_resources.get_mut().drain(..) {
                resources.destroy(&self.device);
            }
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
        Ok(VkContext::with_create_info(VkContextCreateInfo {
            // Android shell runners cannot discover validation layers packaged in an APK.
            with_validation: cfg!(test) && !cfg!(target_os = "android"),
            ..Default::default()
        })?)
    }

    fn device_name(&self) -> Option<&str> {
        self.physical_device.properties.device_name_as_c_str().ok()?.to_str().ok()
    }

    fn create_command_buffer(
        &self,
        _name: Option<&str>,
        allocation_pool: Option<Arc<<Vulkan as Backend>::AllocationPool>>,
    ) -> Result<VkCommandBufferEncoding, Error> {
        let allocation_pool = allocation_pool.unwrap_or_else(|| self.create_allocation_pool());
        VkCommandBufferEncoding::new(self.arc(), allocation_pool)
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<Arc<VkBuffer>, Error> {
        Ok(Arc::new(VkBuffer::new(self.arc(), size as u64)?))
    }

    fn create_sparse_buffer(
        &self,
        _capacity: usize,
    ) -> Result<VkSparseBuffer, Error> {
        Err(Error::NotSupported)
    }

    fn create_allocation_pool(&self) -> Arc<<Vulkan as Backend>::AllocationPool> {
        PoolAllocator::new()
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

    fn device_capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::empty()
    }
}
fn create_instance(
    entry: &ash::Entry,
    api_version: u32,
    create_info: &VkContextCreateInfo,
) -> Result<ash::Instance, VkContextError> {
    let mut instance_extensions: Vec<*const c_char> = Vec::new();
    instance_extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr());

    let mut instance_layers: Vec<*const c_char> = Vec::new();
    let mut instance_nexts = Vec::new();

    if create_info.with_validation {
        if !is_layer_supported(entry, VK_LAYER_KHRONOS_VALIDATION)? {
            return Err(VkContextError::ValidationNotSupported);
        }
        instance_extensions.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
        instance_layers.push(VK_LAYER_KHRONOS_VALIDATION.as_ptr());

        let msg_create_info = debug_messenger_info(create_info);
        instance_nexts.push(msg_create_info);
    }

    let app_info = vk::ApplicationInfo::default().api_version(api_version);

    let mut instance_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);
    for next in instance_nexts.iter_mut() {
        instance_info = instance_info.push_next(next);
    }

    unsafe { entry.create_instance(&instance_info, None) }.map_err(VkContextError::InstanceCreate)
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
    required_extensions: &[&CStr],
    required_features: VkPhysicalDeviceFeatures,
) -> Result<VkPhysicalDevice, VkContextError> {
    let devices = unsafe { instance.enumerate_physical_devices() }.map_err(VkContextError::PhysicalDevicesNotFound)?;

    let device_opt = devices
        .into_iter()
        .map(|device| VkPhysicalDevice::new(entry, instance, device))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|physical_device| {
            physical_device.properties.api_version >= vk::API_VERSION_1_3
                && physical_device.features.contains(required_features)
                && physical_device.capabilities.contains(VkDeviceCapabilities::FLOAT_CONTROLS)
                && physical_device.subgroup_properties.supported_operations.contains(
                    vk::SubgroupFeatureFlags::BASIC
                        | vk::SubgroupFeatureFlags::ARITHMETIC
                        | vk::SubgroupFeatureFlags::VOTE
                        | vk::SubgroupFeatureFlags::BALLOT
                        | vk::SubgroupFeatureFlags::SHUFFLE,
                )
                && physical_device.subgroup_properties.supported_stages.contains(vk::ShaderStageFlags::COMPUTE)
                && physical_device
                    .memory_properties
                    .memory_types_as_slice()
                    .iter()
                    .any(|memory_type| memory_type.property_flags.contains(HOST_COHERENT_MEMORY))
                && required_extensions.iter().all(|required| {
                    physical_device.supported_extensions.iter().any(|name| name.as_bytes() == required.to_bytes())
                })
        })
        .min_by_key(|physical_device| match physical_device.properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
            vk::PhysicalDeviceType::CPU => 3,
            vk::PhysicalDeviceType::OTHER => 4,
            _ => 5,
        });
    device_opt.ok_or(VkContextError::PhysicalDeviceSuitableNotFound)
}
fn get_logical_device(
    instance: &ash::Instance,
    physical_device: &VkPhysicalDevice,
    required_extensions: &[&CStr],
    required_features: VkPhysicalDeviceFeatures,
) -> Result<(ash::Device, u32, u32), VkContextError> {
    // A compute queue whose command buffers time themselves.
    let queue_family_properties =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device.device) };
    let queue_family_index = queue_family_properties
        .iter()
        .position(|properties| {
            properties.queue_flags.contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)
                && properties.timestamp_valid_bits > 0
        })
        .ok_or(VkContextError::PhysicalDeviceQueueNotFound)? as u32;

    let queue_priorities = [1.0_f32];
    let queue_create_info =
        vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family_index).queue_priorities(&queue_priorities);

    // prepare extensions
    let capabilities = physical_device.capabilities;
    let bf16_cores = capabilities.contains(VkDeviceCapabilities::BF16_TENSOR_CORES);
    let matrices = capabilities.intersects(
        VkDeviceCapabilities::BF16_TENSOR_CORES
            | VkDeviceCapabilities::FP16_TENSOR_CORES
            | VkDeviceCapabilities::INT8_TENSOR_CORES,
    );
    let element_fold = capabilities.contains(VkDeviceCapabilities::INT8_ELEMENT_FOLD);
    let mut device_extensions = required_extensions.iter().map(|name| name.as_ptr()).collect::<Vec<_>>();
    if capabilities.contains(VkDeviceCapabilities::MEMORY_BUDGET) {
        device_extensions.push(ext::memory_budget::NAME.as_ptr());
    }
    if matrices {
        device_extensions.push(khr::cooperative_matrix::NAME.as_ptr());
    }
    if bf16_cores {
        device_extensions.push(VkShaderBfloat16Features::EXTENSION_NAME.as_ptr());
    }
    if element_fold {
        device_extensions.push(VkCooperativeMatrix2Features::EXTENSION_NAME.as_ptr());
    }

    // prepare features
    let enabled = |feature| required_features.contains(feature);
    let vk10_features =
        vk::PhysicalDeviceFeatures::default().shader_int16(enabled(VkPhysicalDeviceFeatures::SHADER_INT16));
    let mut vk11_features = vk::PhysicalDeviceVulkan11Features::default()
        .storage_buffer16_bit_access(enabled(VkPhysicalDeviceFeatures::STORAGE_BUFFER_16BIT_ACCESS))
        .storage_push_constant16(enabled(VkPhysicalDeviceFeatures::STORAGE_PUSH_CONSTANT16));
    let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default()
        .shader_int8(enabled(VkPhysicalDeviceFeatures::SHADER_INT8))
        .storage_buffer8_bit_access(enabled(VkPhysicalDeviceFeatures::STORAGE_BUFFER_8BIT_ACCESS))
        .storage_push_constant8(enabled(VkPhysicalDeviceFeatures::STORAGE_PUSH_CONSTANT8))
        .shader_float16(enabled(VkPhysicalDeviceFeatures::SHADER_FLOAT16))
        .shader_subgroup_extended_types(enabled(VkPhysicalDeviceFeatures::SHADER_SUBGROUP_EXTENDED_TYPES))
        .buffer_device_address(enabled(VkPhysicalDeviceFeatures::BUFFER_DEVICE_ADDRESS))
        .host_query_reset(enabled(VkPhysicalDeviceFeatures::HOST_QUERY_RESET))
        .vulkan_memory_model(enabled(VkPhysicalDeviceFeatures::VULKAN_MEMORY_MODEL))
        .vulkan_memory_model_device_scope(enabled(VkPhysicalDeviceFeatures::VULKAN_MEMORY_MODEL_DEVICE_SCOPE));
    let mut vk13_features = vk::PhysicalDeviceVulkan13Features::default()
        .maintenance4(enabled(VkPhysicalDeviceFeatures::MAINTENANCE4))
        .synchronization2(enabled(VkPhysicalDeviceFeatures::SYNCHRONIZATION2))
        .subgroup_size_control(
            physical_device.subgroup_properties.required_size.is_some()
                || capabilities.contains(VkDeviceCapabilities::GEMV_INT8_NATIVE_128),
        )
        .compute_full_subgroups(capabilities.contains(VkDeviceCapabilities::FULL_SUBGROUPS))
        .shader_integer_dot_product(capabilities.contains(VkDeviceCapabilities::INTEGER_DOT_PRODUCT));

    // prepare device
    let mut matrix_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
    let mut bfloat16_features = VkShaderBfloat16Features::default();
    bfloat16_features.shader_bfloat16_type = vk::TRUE;
    bfloat16_features.shader_bfloat16_cooperative_matrix = vk::TRUE;
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
    if bf16_cores {
        device_create_info = device_create_info.push_next(&mut bfloat16_features);
    }
    if element_fold {
        device_create_info = device_create_info.push_next(&mut matrix2_features);
    }
    let device = unsafe { instance.create_device(physical_device.device, &device_create_info, None) }
        .map_err(VkContextError::DeviceCreateError)?;

    Ok((device, queue_family_index, queue_family_properties[queue_family_index as usize].timestamp_valid_bits))
}
fn is_layer_supported(
    entry: &ash::Entry,
    layer: &CStr,
) -> Result<bool, VkContextError> {
    let layer_properties = unsafe { entry.enumerate_instance_layer_properties() }?;
    Ok(layer_properties.iter().any(|properties| unsafe { CStr::from_ptr(properties.layer_name.as_ptr()) } == layer))
}

fn debug_messenger_info(create_info: &VkContextCreateInfo) -> vk::DebugUtilsMessengerCreateInfoEXT<'_> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        .pfn_user_callback(Some(debug_message_callback))
        .user_data(create_info as *const VkContextCreateInfo as *mut c_void)
}

unsafe extern "system" fn debug_message_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    p_user_data: *mut c_void,
) -> vk::Bool32 {
    if p_user_data.is_null() {
        return vk::FALSE;
    }

    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "",
    };
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
    let log_message = format!("{types}{:?}", message);

    let create_info = unsafe { &*(p_user_data as *const VkContextCreateInfo) };
    let logger = create_info.logger.as_ref();
    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => logger.v(log_message.as_str()),
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => logger.i(log_message.as_str()),
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => logger.w(log_message.as_str()),
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => logger.e(log_message.as_str()),
        _ => logger.d(log_message.as_str()),
    }

    vk::FALSE
}
