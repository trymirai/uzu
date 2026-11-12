#[cfg(not(target_arch = "wasm32"))]
use std::thread;
use std::{
    borrow::Cow,
    collections::{HashMap, hash_map::Entry},
    convert::Infallible,
    path::Path,
    sync::{Arc, Weak},
};

use futures::channel::oneshot;
use parking_lot::Mutex;

use crate::backends::{
    common::{Backend, BufferRef, Context, DeviceCapabilities, allocator::pool::PoolAllocator},
    webgpu::{
        WebGPU,
        buffer::dense::{WebGPUDenseBufferCpuGpu, WebGPUDenseBufferGpu},
        command_buffer::WebGPUCommandBufferEncoding,
        error::WebGPUError,
    },
};

#[derive(PartialEq, Eq, Hash)]
struct PipelineCacheKey {
    shader_module: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    constants: Vec<(&'static str, [u8; 8])>,
    entry_point: &'static str,
}

pub struct WebGPUContext {
    pipeline_cache: Mutex<HashMap<PipelineCacheKey, wgpu::ComputePipeline>>,
    pipeline_layout_cache: Mutex<HashMap<wgpu::BindGroupLayout, wgpu::PipelineLayout>>,
    bind_group_layout_cache: Mutex<HashMap<Vec<wgpu::BindGroupLayoutEntry>, wgpu::BindGroupLayout>>,
    shader_module_cache: Mutex<HashMap<&'static str, wgpu::ShaderModule>>,
    pub(super) queue: wgpu::Queue,
    pub(super) limits: wgpu::Limits,
    pub(super) device: wgpu::Device,
    _adapter: wgpu::Adapter,
    _instance: wgpu::Instance,
    device_name: String,
    weak_self: Weak<WebGPUContext>,
}

impl WebGPUContext {
    pub(super) fn get_shader_module(
        &self,
        source: &'static str,
    ) -> wgpu::ShaderModule {
        match self.shader_module_cache.lock().entry(source) {
            Entry::Occupied(occupied) => occupied.get().clone(),
            Entry::Vacant(vacant) => {
                let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
                });
                vacant.insert(module).clone()
            },
        }
    }

    pub(super) fn get_bind_group_layout(
        &self,
        entries: impl IntoIterator<Item = wgpu::BindGroupLayoutEntry>,
    ) -> wgpu::BindGroupLayout {
        let entries = entries.into_iter().collect::<Vec<wgpu::BindGroupLayoutEntry>>();

        match self.bind_group_layout_cache.lock().entry(entries) {
            Entry::Occupied(occupied) => occupied.get().clone(),
            Entry::Vacant(vacant) => {
                let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: vacant.key(),
                });
                vacant.insert(bind_group_layout).clone()
            },
        }
    }

    pub(super) fn get_pipeline_layout(
        &self,
        bind_group_layout: wgpu::BindGroupLayout,
    ) -> wgpu::PipelineLayout {
        match self.pipeline_layout_cache.lock().entry(bind_group_layout) {
            Entry::Occupied(occupied) => occupied.get().clone(),
            Entry::Vacant(vacant) => {
                let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(vacant.key())],
                    immediate_size: 0,
                });
                vacant.insert(pipeline_layout).clone()
            },
        }
    }

    pub(super) fn get_pipeline(
        &self,
        shader_module: wgpu::ShaderModule,
        pipeline_layout: wgpu::PipelineLayout,
        constants: Vec<(&'static str, f64)>,
        entry_point: &'static str,
    ) -> wgpu::ComputePipeline {
        let constants =
            constants.into_iter().map(|(k, v)| (k, bytemuck::cast(v))).collect::<Vec<(&'static str, [u8; 8])>>();

        match self.pipeline_cache.lock().entry(PipelineCacheKey {
            shader_module,
            pipeline_layout,
            constants,
            entry_point,
        }) {
            Entry::Occupied(occupied) => occupied.get().clone(),
            Entry::Vacant(vacant) => {
                let key = vacant.key();

                let constants =
                    key.constants.iter().map(|(k, v)| (*k, bytemuck::cast(*v))).collect::<Vec<(&'static str, f64)>>();

                let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&key.pipeline_layout),
                    module: &key.shader_module,
                    entry_point: Some(key.entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        zero_initialize_workgroup_memory: true, // required on non-native
                    },
                    cache: None,
                });
                vacant.insert(pipeline).clone()
            },
        }
    }
}

impl Context for WebGPUContext {
    type Backend = WebGPU;

    async fn new() -> Result<Arc<Self>, WebGPUError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await?;
        if !adapter.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::SHADER_F16_IN_F32) {
            return Err(WebGPUError::NotSupported);
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SHADER_F16
                    | wgpu::Features::SUBGROUP
                    | wgpu::Features::SUBGROUP_BARRIER,
                required_limits: adapter.limits(), // TODO: specify actual limits
                ..Default::default()
            })
            .await?;

        Ok(Arc::new_cyclic(|weak_self| WebGPUContext {
            pipeline_cache: Mutex::new(HashMap::new()),
            pipeline_layout_cache: Mutex::new(HashMap::new()),
            bind_group_layout_cache: Mutex::new(HashMap::new()),
            shader_module_cache: Mutex::new(HashMap::new()),
            queue,
            limits: device.limits(),
            device,
            device_name: adapter.get_info().name,
            _adapter: adapter,
            _instance: instance,
            weak_self: weak_self.clone(),
        }))
    }

    fn device_name(&self) -> Option<&str> {
        Some(&self.device_name)
    }

    fn create_command_buffer(
        &self,
        name: Option<&str>,
        allocation_pool: Option<Arc<<WebGPU as Backend>::AllocationPool>>,
    ) -> Result<WebGPUCommandBufferEncoding, WebGPUError> {
        Ok(WebGPUCommandBufferEncoding::new(
            self.weak_self.upgrade().unwrap(),
            name,
            allocation_pool.unwrap_or_else(|| self.create_allocation_pool()),
        ))
    }

    fn create_buffer_gpu(
        &self,
        size: usize,
    ) -> Result<<WebGPU as Backend>::GlobalGpuBuffer, WebGPUError> {
        WebGPUDenseBufferGpu::new(self, size)
    }

    fn create_buffer_cpu_gpu(
        &self,
        size: usize,
    ) -> Result<<WebGPU as Backend>::GlobalBufferCpuGpu, WebGPUError> {
        WebGPUDenseBufferCpuGpu::new(self, size)
    }

    fn copyout<T: bytemuck::AnyBitPattern>(
        &self,
        buffer: impl BufferRef<Backend = WebGPU, Buffer = WebGPUDenseBufferGpu>,
    ) -> impl Future<Output = Result<Vec<T>, <Self::Backend as Backend>::Error>> + Send {
        let (buffer, buffer_range) = buffer.parts();

        let cpu_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer_range.iter().len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut command_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        command_encoder.copy_buffer_to_buffer(
            buffer.wgpu_buffer(),
            buffer_range.start as u64,
            &cpu_buffer,
            0,
            buffer_range.iter().len() as u64,
        );
        let submission_index = self.queue.submit([command_encoder.finish()]);

        let (sender, receiver) = oneshot::channel();
        cpu_buffer.map_async(wgpu::MapMode::Read, .., move |err| {
            let _ = sender.send(err);
        });

        #[cfg(not(target_arch = "wasm32"))]
        let context = self.weak_self.upgrade().unwrap();
        #[cfg(not(target_arch = "wasm32"))]
        thread::spawn(move || {
            let _ = context.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission_index),
                timeout: None,
            });
        });

        #[cfg(target_arch = "wasm32")]
        let _ = submission_index;

        async move {
            receiver.await.map_err(|_| WebGPUError::CommandBuffer)??;
            Ok(bytemuck::cast_slice(&cpu_buffer.get_mapped_range(..)?).to_vec())
        }
    }

    fn create_allocation_pool(&self) -> Arc<<WebGPU as Backend>::AllocationPool> {
        PoolAllocator::new()
    }

    fn create_sparse_buffer(
        &self,
        _capacity: usize,
    ) -> Result<<Self::Backend as Backend>::SparseBuffer, <Self::Backend as Backend>::Error> {
        Err(WebGPUError::NotSupported)
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        None
    }

    fn start_capture(
        &self,
        _trace_path: &Path,
    ) -> Result<Infallible, WebGPUError> {
        Err(WebGPUError::NotSupported)
    }

    fn device_capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::empty()
    }
}
