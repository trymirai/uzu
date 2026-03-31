use std::{ops::Range, time::Duration};

use crate::backends::common::{
    AccessFlags, Allocation, AllocationPool, AllocationType, Backend, BufferGpuAddressRangeExt, CommandBuffer,
    CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable, CommandBufferInitial, CommandBufferPending,
    Context,
    hazard_tracker::{Access, HazardTracker},
};

pub struct Encoder<'encoding, B: Backend> {
    context: &'encoding B::Context,
    command_buffer: <B::CommandBuffer as CommandBuffer>::Encoding,
    allocation_pool: AllocationPool<B>,
    hazard_tracker: HazardTracker,
}

impl<'encoding, B: Backend> Encoder<'encoding, B> {
    pub fn new(context: &'encoding B::Context) -> Result<Self, B::Error> {
        let command_buffer = context.create_command_buffer()?.start_encoding();
        let allocation_pool = context.create_allocation_pool(false);
        let hazard_tracker = HazardTracker::new();

        Ok(Self {
            context,
            command_buffer,
            allocation_pool,
            hazard_tracker,
        })
    }

    // This is valid on both cpu and gpu timelines
    pub fn allocate_constant(
        &mut self,
        size: usize,
    ) -> Result<Allocation<B>, B::Error> {
        self.context.create_allocation(
            size,
            AllocationType::Pooled {
                pool: &self.allocation_pool,
                cpu_available: true,
            },
        )
    }

    // This is valid on gpu timeline only
    pub fn allocate_scratch(
        &mut self,
        size: usize,
    ) -> Result<Allocation<B>, B::Error> {
        self.context.create_allocation(
            size,
            AllocationType::Pooled {
                pool: &self.allocation_pool,
                cpu_available: false,
            },
        )
    }

    pub fn encode_copy(
        &mut self,
        src: &B::Buffer,
        src_range: Range<usize>,
        dst: &mut B::Buffer,
        dst_range: Range<usize>,
    ) {
        let size = src_range.end - src_range.start;
        debug_assert_eq!(size, dst_range.end - dst_range.start);
        self.access(&[
            Access {
                range: src.gpu_address_subrange(src_range.clone()),
                flags: AccessFlags::copy_read(),
            },
            Access {
                range: dst.gpu_address_subrange(dst_range.clone()),
                flags: AccessFlags::copy_write(),
            },
        ]);
        self.command_buffer.encode_copy(src, src_range, dst, dst_range);
    }

    pub fn encode_fill(
        &mut self,
        dst: &mut B::Buffer,
        range: Range<usize>,
        value: u8,
    ) {
        self.access(&[Access {
            range: dst.gpu_address_subrange(range.clone()),
            flags: AccessFlags::copy_write(),
        }]);
        self.command_buffer.encode_fill(dst, range, value);
    }

    pub fn access(
        &mut self,
        accesses: &[Access],
    ) {
        if let Some((after, before)) = self.hazard_tracker.access(accesses) {
            self.command_buffer.encode_barrier(after, before);
        }
    }

    pub fn encode_wait_for_event(
        &mut self,
        event: &B::Event,
        value: u64,
    ) {
        self.command_buffer.encode_wait_for_event(event, value);
    }

    pub fn encode_signal_event(
        &mut self,
        event: &B::Event,
        value: u64,
    ) {
        self.command_buffer.encode_signal_event(event, value);
    }

    pub fn add_completion_handler(
        &mut self,
        handler: impl FnOnce(Result<&<B::CommandBuffer as CommandBuffer>::Completed, B::Error>) + Send + 'static,
    ) {
        self.command_buffer.add_completion_handler(handler);
    }

    pub fn as_command_buffer_mut(&mut self) -> &mut <B::CommandBuffer as CommandBuffer>::Encoding {
        &mut self.command_buffer
    }

    pub fn end_encoding(self) -> Executable<B> {
        Executable {
            command_buffer: self.command_buffer.end_encoding(),
            allocation_pool: self.allocation_pool,
        }
    }
}

pub struct Executable<B: Backend> {
    command_buffer: <B::CommandBuffer as CommandBuffer>::Executable,
    allocation_pool: AllocationPool<B>,
}

impl<B: Backend> Executable<B> {
    pub fn submit(self) -> Pending<B> {
        Pending {
            command_buffer: self.command_buffer.submit(),
            allocation_pool: self.allocation_pool,
        }
    }
}

pub struct Pending<B: Backend> {
    command_buffer: <B::CommandBuffer as CommandBuffer>::Pending,
    allocation_pool: AllocationPool<B>,
}

impl<B: Backend> Pending<B> {
    pub fn wait_until_completed(self) -> Result<Completed<B>, B::Error> {
        Ok(Completed {
            command_buffer: self.command_buffer.wait_until_completed()?,
            allocation_pool: self.allocation_pool,
        })
    }
}

pub struct Completed<B: Backend> {
    command_buffer: <B::CommandBuffer as CommandBuffer>::Completed,
    #[allow(unused)]
    allocation_pool: AllocationPool<B>,
}

impl<B: Backend> Completed<B> {
    pub fn gpu_execution_time(&self) -> Option<Duration> {
        self.command_buffer.gpu_execution_time()
    }
}
