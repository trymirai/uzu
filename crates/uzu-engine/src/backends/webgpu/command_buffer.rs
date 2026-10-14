use std::{collections::HashMap, sync::Arc, time::Duration};

use super::{
    WebGpu, WebGpuError,
    buffer::{BufferStorage, WebGpuBuffer, WebGpuBufferExt},
    context::WebGpuContext,
};
use crate::backends::common::{
    BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
    CommandBufferPending, Context,
    allocator::{
        bump::{BumpAllocation, BumpAllocator},
        pool::{PoolAllocation, PoolAllocator},
    },
};

#[derive(Clone)]
pub struct Binding {
    storage: Arc<BufferStorage>,
    /// Byte offset relative to the entire bound GPU buffer.
    pub offset: u32,
    pub size: u32,
    pub writable: bool,
}

impl Binding {
    pub fn read(buffer: impl BufferRef<Backend = WebGpu>) -> Self {
        let (buffer, range) = buffer.parts();
        let (storage, base) = buffer.gpu_storage();
        Self {
            storage,
            offset: (base + range.start) as u32,
            size: range.iter().len() as u32,
            writable: false,
        }
    }

    pub fn write(buffer: impl BufferMut<Backend = WebGpu>) -> Self {
        let (buffer, range) = buffer.parts();
        let (storage, base) = buffer.gpu_storage();
        Self {
            storage,
            offset: (base + range.start) as u32,
            size: range.iter().len() as u32,
            writable: true,
        }
    }

    pub fn id(&self) -> u32 {
        self.storage.id
    }
    pub fn is_writable(&self) -> bool {
        self.writable
    }
}

// Packed u32 protocol consumed by decodeCommands in bridge.js. Parameters stay
// as their original bit patterns; no JSON number formatting or object cloning
// is needed between the engine and GPU workers.
const DISPATCH: u32 = 1;
const COPY: u32 = 2;
const FILL: u32 = 3;

pub struct WebGpuCommandBuffer;
impl CommandBuffer for WebGpuCommandBuffer {
    type Backend = WebGpu;
    type Encoding = WebGpuCommandBufferEncoding;
    type Executable = WebGpuCommandBufferExecutable;
    type Pending = WebGpuCommandBufferPending;
    type Completed = WebGpuCommandBufferCompleted;
}

pub struct WebGpuCommandBufferEncoding {
    profile_started: Option<f64>,
    commands: Vec<u32>,
    resources: HashMap<u32, (Arc<BufferStorage>, bool)>,
    constant_allocator: BumpAllocator<WebGpuBuffer>,
    allocation_pool: Arc<PoolAllocator<WebGpuBuffer>>,
    context: Arc<WebGpuContext>,
}

impl WebGpuCommandBufferEncoding {
    pub(super) fn new(
        context: Arc<WebGpuContext>,
        constant_allocator: BumpAllocator<WebGpuBuffer>,
        allocation_pool: Arc<PoolAllocator<WebGpuBuffer>>,
    ) -> Self {
        Self {
            profile_started: context.runtime.profile_start(),
            commands: Vec::new(),
            resources: HashMap::new(),
            constant_allocator,
            allocation_pool,
            context,
        }
    }

    fn retain(
        &mut self,
        binding: &Binding,
    ) {
        assert_eq!(binding.storage.runtime.id, self.context.runtime.id, "WebGPU buffers belong to different devices");
        self.resources
            .entry(binding.id())
            .and_modify(|(_, writable)| *writable |= binding.writable)
            .or_insert_with(|| (binding.storage.clone(), binding.writable));
    }

    /// Storage bindings start at zero. The final binding is a uniform buffer
    /// containing `parameters`, padded to sixteen bytes. Entry point: `main`.
    pub fn dispatch(
        &mut self,
        source: &str,
        bindings: Vec<Binding>,
        parameters: &[u32],
        workgroups: [u32; 3],
    ) {
        if workgroups.contains(&0) {
            return;
        }
        let pipeline = self.context.runtime.pipeline(source).expect("WebGPU shader registration failed");
        self.dispatch_pipeline(pipeline, bindings, parameters, workgroups);
    }

    pub fn dispatch_cached(
        &mut self,
        source: &Arc<str>,
        bindings: Vec<Binding>,
        parameters: &[u32],
        workgroups: [u32; 3],
    ) {
        if workgroups.contains(&0) {
            return;
        }
        let pipeline = self.context.runtime.pipeline_cached(source).expect("WebGPU shader registration failed");
        self.dispatch_pipeline(pipeline, bindings, parameters, workgroups);
    }

    fn dispatch_pipeline(
        &mut self,
        pipeline: u32,
        bindings: Vec<Binding>,
        parameters: &[u32],
        workgroups: [u32; 3],
    ) {
        for binding in &bindings {
            self.retain(binding);
        }
        let parameter_count = parameters.len().max(1).next_multiple_of(4);
        self.commands.extend([DISPATCH, pipeline, bindings.len() as u32, parameter_count as u32]);
        self.commands.extend(workgroups);
        self.commands.extend(bindings.iter().map(Binding::id));
        self.commands.extend(parameters);
        self.commands.resize(self.commands.len() + parameter_count - parameters.len(), 0);
    }
}

impl CommandBufferEncoding for WebGpuCommandBufferEncoding {
    type CommandBuffer = WebGpuCommandBuffer;
    fn context(&self) -> &WebGpuContext {
        &self.context
    }

    fn allocate_constant(
        &mut self,
        size: usize,
    ) -> Result<BumpAllocation<WebGpuBuffer>, WebGpuError> {
        self.constant_allocator.allocate(size, |size| self.context.create_buffer(size))
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<PoolAllocation<WebGpuBuffer>, WebGpuError> {
        self.allocation_pool.allocate(size, |size| self.context.create_buffer(size))
    }

    fn encode_copy(
        &mut self,
        src: impl BufferRef<Backend = WebGpu>,
        dst: impl BufferMut<Backend = WebGpu>,
    ) {
        let source = Binding::read(src);
        let destination = Binding::write(dst);
        assert_eq!(source.size, destination.size, "WebGPU copy sizes differ");
        if source.size == 0 {
            return;
        }
        self.retain(&source);
        self.retain(&destination);
        self.commands.extend([COPY, source.id(), source.offset, destination.id(), destination.offset, source.size]);
    }

    fn encode_fill(
        &mut self,
        dst: impl BufferMut<Backend = WebGpu>,
        value: u8,
    ) {
        let destination = Binding::write(dst);
        if destination.size == 0 {
            return;
        }
        self.retain(&destination);
        self.commands.extend([FILL, destination.id(), destination.offset, destination.size, value.into()]);
    }

    fn debug_group(
        &mut self,
        _name: &str,
    ) -> &mut Self {
        self
    }
    fn debug_point(
        &mut self,
        _name: &str,
    ) {
    }

    fn end_encoding(self) -> WebGpuCommandBufferExecutable {
        assert!(self.constant_allocator.is_done(), "attempted to end encoding while constants are still alive");
        self.context.runtime.profile_end("engineEncodeMs", self.profile_started);
        WebGpuCommandBufferExecutable(self)
    }
}

pub struct WebGpuCommandBufferExecutable(WebGpuCommandBufferEncoding);
impl CommandBufferExecutable for WebGpuCommandBufferExecutable {
    type CommandBuffer = WebGpuCommandBuffer;

    fn submit(self) -> WebGpuCommandBufferPending {
        let started = self.0.context.runtime.profile_start();
        let result = (|| {
            let mut written_buffers = Vec::new();
            for (buffer, writable) in self.0.resources.values() {
                buffer.ensure_uploaded()?;
                if *writable {
                    written_buffers.push(buffer.id);
                }
            }
            let submission = self.0.context.runtime.submit(&self.0.commands, &written_buffers)?;
            for (buffer, writable) in self.0.resources.values() {
                if *writable {
                    buffer.mark_gpu_written();
                }
            }
            Ok(submission)
        })();
        self.0.context.runtime.profile_end("engineSubmitMs", started);
        WebGpuCommandBufferPending {
            submission: result,
            encoding: self.0,
        }
    }
}

pub struct WebGpuCommandBufferPending {
    submission: Result<u32, WebGpuError>,
    // Retains resources until completion. GPU-only scratch allocations never
    // acquire CPU mirrors or get read back as a side effect of waiting.
    encoding: WebGpuCommandBufferEncoding,
}
impl CommandBufferPending for WebGpuCommandBufferPending {
    type CommandBuffer = WebGpuCommandBuffer;

    fn wait_until_completed(self) -> Result<WebGpuCommandBufferCompleted, WebGpuError> {
        self.encoding.context.runtime.wait(self.submission?)?;
        Ok(WebGpuCommandBufferCompleted)
    }
}

pub struct WebGpuCommandBufferCompleted;
impl CommandBufferCompleted for WebGpuCommandBufferCompleted {
    fn gpu_execution_time(&self) -> Duration {
        // Queue completion wall time includes IPC and is not GPU execution time.
        // Report no device time until timestamp queries are implemented.
        Duration::ZERO
    }
    type CommandBuffer = WebGpuCommandBuffer;
}
