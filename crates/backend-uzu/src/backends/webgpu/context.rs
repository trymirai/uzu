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
    shader_module_cache: RefCell<HashMap<&'static str, wgpu::ShaderModule>>,
    allocator: Rc<Allocator<WebGPU>>,
    pub(super) queue: wgpu::Queue,
    device: wgpu::Device,
    adapter: wgpu::Adapter,
    instance: wgpu::Instance,
    weak_self: Weak<WebGPUContext>,
}

impl WebGPUContext {
    pub fn get_shader_module(
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
}

impl Context for WebGPUContext {
    type Backend = WebGPU;

    fn new() -> Result<Rc<Self>, WebGPUError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;

        Ok(Rc::new_cyclic(|weak_self| WebGPUContext {
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
        8
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
                | wgpu::BufferUsages::INDIRECT,
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
        todo!()
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
