use std::{
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use crate::{
    backends::{
        common::{
            Backend, BlockName, BufferMut, BufferRef, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding,
            CommandBufferExecutable, CommandBufferPending, CommandBufferTimestamps, TimestampSpan,
            TimestampSpanRecorder, allocator::bump::BumpAllocator,
        },
        cpu::{
            Cpu,
            buffer::{CpuBufferExt, dense::CpuBuffer},
            context::CpuContext,
            cpu_command::CpuCommand,
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
    commands: Vec<CpuCommand>,
    constant_allocator: BumpAllocator<<Cpu as Backend>::GlobalBuffer>,
    allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
    context: Arc<CpuContext>,
    timestamp_spans: Option<TimestampSpanRecorder>,
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
            timestamp_spans: None,
        }
    }

    pub fn push_command(
        &mut self,
        command: impl FnOnce() + Send + 'static,
    ) {
        self.commands.push(CpuCommand::Run(Box::new(command)))
    }

    fn write_timestamp(&mut self) {
        self.commands.push(CpuCommand::Timestamp);
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

    fn enable_timestamps(&mut self) {
        assert!(self.timestamp_spans.is_none(), "timestamps already enabled");
        self.timestamp_spans = Some(TimestampSpanRecorder::default());
    }

    fn sample_start_timestamp(
        &mut self,
        name: &BlockName,
    ) {
        if let Some(spans) = &mut self.timestamp_spans {
            spans.start(name.clone());
            self.write_timestamp();
        }
    }

    fn sample_end_timestamp(&mut self) {
        if let Some(spans) = &mut self.timestamp_spans {
            spans.end();
            self.write_timestamp();
        }
    }

    fn end_encoding(self) -> CpuCommandBufferExecutable {
        assert!(self.constant_allocator.is_done(), "attempted to end encoding while constants are still alive");
        CpuCommandBufferExecutable {
            commands: self.commands,
            constant_allocator: self.constant_allocator,
            allocation_pool: self.allocation_pool,
            context: self.context,
            timestamp_spans: self.timestamp_spans,
        }
    }
}

pub struct CpuCommandBufferExecutable {
    commands: Vec<CpuCommand>,
    constant_allocator: BumpAllocator<<Cpu as Backend>::GlobalBuffer>,
    allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
    context: Arc<CpuContext>,
    timestamp_spans: Option<TimestampSpanRecorder>,
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

                let instants = self
                    .commands
                    .into_iter()
                    .filter_map(|command| match command {
                        CpuCommand::Run(run) => {
                            run();
                            None
                        },
                        CpuCommand::Timestamp => Some(Instant::now()),
                    })
                    .collect::<Box<[Instant]>>();

                let gpu_execution_time = start.elapsed();

                let completed = CpuCommandBufferCompleted {
                    gpu_execution_time,
                    timestamps: self.timestamp_spans.map_or_else(Box::default, |spans| spans.into_spans(&instants)),
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
    timestamps: CommandBufferTimestamps,
    _allocation_pool: Arc<<Cpu as Backend>::AllocationPool>,
}

impl CommandBufferCompleted for CpuCommandBufferCompleted {
    type CommandBuffer = CpuCommandBuffer;

    fn gpu_execution_time(&self) -> Duration {
        self.gpu_execution_time
    }

    fn timestamps(&self) -> &[TimestampSpan] {
        &self.timestamps
    }
}
