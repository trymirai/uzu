use std::{
    mem::take,
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use crate::{
    backends::{
        common::{
            Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding,
            CommandBufferExecutable, CommandBufferPending, TimestampSampleEntry, allocator::bump::BumpAllocator,
        },
        cpu::{
            Cpu,
            buffer::{CpuBufferExt, dense::CpuBuffer},
            context::CpuContext,
            error::CpuError,
        },
    },
    utils::pointers::{SendPtr, SendPtrMut},
};

pub struct CpuCommandBuffer;

impl CommandBuffer for CpuCommandBuffer {
    type Backend = Cpu;

    type Encoding = CpuCommandBufferEncoding;
    type Executable = CpuCommandBufferExecutable;
    type Pending = CpuCommandBufferPending;
    type Completed = CpuCommandBufferCompleted;
}

pub struct CpuCommandBufferEncoding {
    commands: Vec<Box<dyn FnOnce() + Send>>,
    constant_allocator: BumpAllocator<<Cpu as Backend>::GlobalBuffer>,
    allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
    context: Arc<CpuContext>,
    timestamps: Option<Arc<Mutex<Vec<(TimestampSampleEntry, Instant)>>>>,
}

impl CpuCommandBufferEncoding {
    pub fn new(
        constant_allocator: BumpAllocator<<Cpu as Backend>::GlobalBuffer>,
        allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
        context: Arc<CpuContext>,
    ) -> CpuCommandBufferEncoding {
        CpuCommandBufferEncoding {
            commands: Vec::new(),
            constant_allocator,
            allocation_pool,
            context,
            timestamps: None,
        }
    }

    pub fn push_command(
        &mut self,
        command: impl FnOnce() + Send + 'static,
    ) {
        self.commands.push(Box::new(command))
    }

    fn sample_timestamp(
        &mut self,
        entry: fn(String) -> TimestampSampleEntry,
        name: &str,
    ) {
        let Some(timestamps) = self.timestamps.clone() else {
            return;
        };
        let entry = entry(name.to_string());
        self.push_command(move || timestamps.lock().push((entry, Instant::now())));
    }
}

impl CommandBufferEncoding for CpuCommandBufferEncoding {
    type CommandBuffer = CpuCommandBuffer;

    fn context(&self) -> &CpuContext {
        &self.context
    }

    fn allocate_constant(
        &mut self,
        size: usize,
    ) -> Result<<Cpu as Backend>::ConstantBuffer, CpuError> {
        self.constant_allocator.allocate(size, |size| Ok(CpuBuffer::new(size)))
    }

    fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<<Cpu as Backend>::ScratchBuffer, CpuError> {
        self.allocation_pool.allocate(size, |size| Ok(CpuBuffer::new(size)))
    }

    fn encode_copy(
        &mut self,
        src: impl BufferRef<Backend = Cpu>,
        dst: impl BufferMut<Backend = Cpu>,
    ) {
        let (src, src_range) = src.parts();
        let (dst, dst_range) = dst.parts();
        assert_eq!(src_range.iter().len(), dst_range.iter().len());

        let src_ptr = SendPtr(unsafe { src.cpu_address().as_ptr().cast::<u8>().add(src_range.start) });
        let dst_ptr = SendPtrMut(unsafe { dst.cpu_address().as_ptr().cast::<u8>().add(dst_range.start) });

        self.push_command(move || unsafe {
            std::ptr::copy(src_ptr.as_ptr(), dst_ptr.as_ptr(), src_range.iter().len());
        });
    }

    fn encode_fill(
        &mut self,
        dst: impl BufferMut<Backend = Cpu>,
        value: u8,
    ) {
        let (dst, range) = dst.parts();
        let size = range.iter().len();
        let dst = SendPtrMut(unsafe { dst.cpu_address().as_ptr().cast::<u8>().add(range.start) });
        self.push_command(move || unsafe {
            dst.as_ptr().write_bytes(value, size);
        });
    }

    fn push_debug_group(
        &mut self,
        _name: &str,
    ) {
    }

    fn pop_debug_group(&mut self) {}

    fn enable_timestamps(&mut self) -> Result<(), CpuError> {
        assert!(self.timestamps.is_none(), "timestamps already enabled");
        self.timestamps = Some(Arc::new(Mutex::new(Vec::new())));
        Ok(())
    }

    fn sample_start_timestamp(
        &mut self,
        name: &String,
    ) {
        self.sample_timestamp(TimestampSampleEntry::Start, name);
    }

    fn sample_end_timestamp(
        &mut self,
        name: &String,
    ) {
        self.sample_timestamp(TimestampSampleEntry::End, name);
    }

    fn end_encoding(self) -> CpuCommandBufferExecutable {
        assert!(self.constant_allocator.is_done(), "attempted to end encoding while constants are still alive");
        CpuCommandBufferExecutable {
            commands: self.commands,
            constant_allocator: self.constant_allocator,
            allocation_pool: self.allocation_pool,
            context: self.context,
            timestamps: self.timestamps,
        }
    }
}

pub struct CpuCommandBufferExecutable {
    commands: Vec<Box<dyn FnOnce() + Send>>,
    constant_allocator: BumpAllocator<<Cpu as Backend>::GlobalBuffer>,
    allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
    context: Arc<CpuContext>,
    timestamps: Option<Arc<Mutex<Vec<(TimestampSampleEntry, Instant)>>>>,
}

impl CommandBufferExecutable for CpuCommandBufferExecutable {
    type CommandBuffer = CpuCommandBuffer;

    fn submit(self) -> CpuCommandBufferPending {
        let (return_sender, return_receiver) = mpsc::channel();
        let allocation_pool = self.allocation_pool.clone();

        self.context
            .command_queue
            .send(Box::new(move || {
                let start = Instant::now();

                for command in self.commands {
                    command()
                }

                let gpu_execution_time = start.elapsed();

                let completed = CpuCommandBufferCompleted {
                    gpu_execution_time,
                    timestamps: self
                        .timestamps
                        .map_or_else(Box::default, |timestamps| take(&mut *timestamps.lock()).into_boxed_slice()),
                    _allocation_pool: self.allocation_pool,
                };

                let _ = return_sender.send(completed);
                drop(self.constant_allocator);
            }))
            .unwrap();

        CpuCommandBufferPending {
            return_receiver,
            _allocation_pool: allocation_pool,
        }
    }
}

pub struct CpuCommandBufferPending {
    return_receiver: mpsc::Receiver<CpuCommandBufferCompleted>,
    _allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
}

impl CommandBufferPending for CpuCommandBufferPending {
    type CommandBuffer = CpuCommandBuffer;

    fn wait_until_completed(self) -> Result<CpuCommandBufferCompleted, CpuError> {
        Ok(self.return_receiver.recv()?)
    }
}

pub struct CpuCommandBufferCompleted {
    gpu_execution_time: Duration,
    timestamps: Box<[(TimestampSampleEntry, Instant)]>,
    _allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
}

impl CommandBufferCompleted for CpuCommandBufferCompleted {
    type CommandBuffer = CpuCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        self.gpu_execution_time
    }

    fn timestamps(&self) -> &[(TimestampSampleEntry, Instant)] {
        &self.timestamps
    }
}
