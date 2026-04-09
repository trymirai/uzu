use std::{
    borrow::Cow,
    cell::RefCell,
    collections::{HashMap, hash_map::Entry},
    path::Path,
    rc::{Rc, Weak},
};

use crate::backends::{
    common::{Allocation, AllocationPool, AllocationType, Allocator, Context},
    webgpu::{
        WebGPU, buffer::WebGPUBuffer, command_buffer::WebGPUCommandBufferInitial, error::WebGPUError,
        event::WebGPUEvent,
    },
};

pub struct WebGPUContext {
    pipeline_cache: RefCell<
        HashMap<
            (wgpu::ShaderModule, wgpu::PipelineLayout, Vec<(&'static str, [u8; 8])>, &'static str),
            wgpu::ComputePipeline,
        >,
    >,
    pipeline_layout_cache: RefCell<HashMap<wgpu::BindGroupLayout, wgpu::PipelineLayout>>,
    bind_group_layout_cache: RefCell<HashMap<Vec<wgpu::BindGroupLayoutEntry>, wgpu::BindGroupLayout>>,
    shader_module_cache: RefCell<HashMap<&'static str, wgpu::ShaderModule>>,
    allocator: Rc<Allocator<WebGPU>>,
    pub(super) queue: wgpu::Queue,
    pub(super) device: wgpu::Device,
    adapter: wgpu::Adapter,
    instance: wgpu::Instance,
    weak_self: Weak<WebGPUContext>,
}

impl WebGPUContext {
    pub(super) fn get_shader_module(
        &self,
        source: &'static str,
    ) -> wgpu::ShaderModule {
        match self.shader_module_cache.borrow_mut().entry(source) {
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

        match self.bind_group_layout_cache.borrow_mut().entry(entries) {
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
        match self.pipeline_layout_cache.borrow_mut().entry(bind_group_layout) {
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

        match self.pipeline_cache.borrow_mut().entry((shader_module, pipeline_layout, constants, entry_point)) {
            Entry::Occupied(occupied) => occupied.get().clone(),
            Entry::Vacant(vacant) => {
                let (shader_module, pipeline_layout, constants, entry_point) = vacant.key();

                let constants =
                    constants.into_iter().map(|(k, v)| (*k, bytemuck::cast(*v))).collect::<Vec<(&'static str, f64)>>();

                let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(pipeline_layout),
                    module: shader_module,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        zero_initialize_workgroup_memory: true,
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

    fn new() -> Result<Rc<Self>, WebGPUError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::default() | wgpu::Features::MAPPABLE_PRIMARY_BUFFERS,
            required_limits: adapter.limits(), // TODO: specify actual limits
            ..Default::default()
        }))?;

        Ok(Rc::new_cyclic(|weak_self| WebGPUContext {
            pipeline_cache: RefCell::new(HashMap::new()),
            pipeline_layout_cache: RefCell::new(HashMap::new()),
            bind_group_layout_cache: RefCell::new(HashMap::new()),
            shader_module_cache: RefCell::new(HashMap::new()),
            allocator: Allocator::new(weak_self.clone()),
            queue,
            device,
            adapter,
            instance,
            weak_self: weak_self.clone(),
        }))
    }

    fn recommended_async_batch_size(
        &self,
        _model_path: &Path,
    ) -> usize {
        0
    }

    fn is_high_performance(&self) -> bool {
        true
    }

    fn debug_active(&self) -> bool {
        false
    }

    fn create_command_buffer(&self) -> Result<WebGPUCommandBufferInitial, WebGPUError> {
        let command_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        Ok(WebGPUCommandBufferInitial {
            command_encoder,
            context: self.weak_self.upgrade().unwrap(),
        })
    }

    fn create_buffer(
        &self,
        size: usize,
    ) -> Result<WebGPUBuffer, WebGPUError> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::MAP_WRITE
                | wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(WebGPUBuffer {
            buffer,
        })
    }

    fn create_allocation(
        &self,
        size: usize,
        allocation_type: AllocationType<WebGPU>,
    ) -> Result<Allocation<WebGPU>, WebGPUError> {
        self.allocator.allocate(size, allocation_type)
    }

    fn create_allocation_pool(
        &self,
        reusable: bool,
    ) -> AllocationPool<WebGPU> {
        self.allocator.create_pool(reusable)
    }

    fn create_event(&self) -> Result<WebGPUEvent, WebGPUError> {
        Ok(WebGPUEvent {})
    }

    fn peak_memory_usage(&self) -> Option<usize> {
        None
    }

    fn enable_capture() {}

    fn start_capture(
        &self,
        _trace_path: &Path,
    ) -> Result<(), WebGPUError> {
        Ok(())
    }

    fn stop_capture(&self) -> Result<(), WebGPUError> {
        Ok(())
    }

    fn tf32_enabled() -> bool {
        false
    }
}
