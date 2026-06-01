use std::{
    ops::{Bound, Range, RangeBounds},
    rc::Rc,
    time::Duration,
};

use crate::backends::common::{
    AccessFlags, Allocation, AllocationPool, AllocationType, AsBufferRangeMut, AsBufferRangeRef, Backend, Buffer,
    BufferGpuAddressRangeExt, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
    CommandBufferInitial, CommandBufferPending, Context,
    hazard_tracker::{Access, HazardTracker},
};

fn resolve_copy_range(
    range: impl RangeBounds<usize>,
    allocation_len: usize,
    label: &str,
) -> Range<usize> {
    let start = match range.start_bound() {
        Bound::Included(&value) => value,
        Bound::Excluded(&value) => value.checked_add(1).expect("copy range start overflow"),
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&value) => value.checked_add(1).expect("copy range end overflow"),
        Bound::Excluded(&value) => value,
        Bound::Unbounded => allocation_len,
    };
    assert!(start <= end, "{label} copy range start exceeds end");
    assert!(end <= allocation_len, "{label} copy range exceeds allocation");
    start..end
}

pub struct Encoder<'encoding, B: Backend> {
    context: &'encoding B::Context,
    command_buffer: <B::CommandBuffer as CommandBuffer>::Encoding,
    allocation_pool: Rc<AllocationPool<B>>,
    hazard_tracker: HazardTracker,
}

impl<'encoding, B: Backend> Encoder<'encoding, B> {
    pub fn new(context: &'encoding B::Context) -> Result<Self, B::Error> {
        Self::new_with_pool(context, Rc::new(context.create_allocation_pool(false)))
    }

    pub fn new_with_pool(
        context: &'encoding B::Context,
        allocation_pool: Rc<AllocationPool<B>>,
    ) -> Result<Self, B::Error> {
        let command_buffer = context.create_command_buffer()?.start_encoding();
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

    pub fn encode_copy<
        Src: AsBufferRangeRef<Buffer: Buffer<Backend = B>>,
        Dst: AsBufferRangeMut<Buffer: Buffer<Backend = B>>,
    >(
        &mut self,
        src: &Src,
        src_range: impl RangeBounds<usize>,
        dst: &mut Dst,
        dst_range: impl RangeBounds<usize>,
    ) {
        let src_buffer_range = src.as_buffer_range_ref();
        let dst_buffer_range = dst.as_buffer_range_mut();
        let src_range = resolve_copy_range(src_range, src_buffer_range.range().len(), "source");
        let dst_range = resolve_copy_range(dst_range, dst_buffer_range.range().len(), "destination");
        let byte_len = src_range.len();
        assert_eq!(byte_len, dst_range.len(), "copy range lengths must match");
        assert!(byte_len > 0, "zero-sized copies are not allowed");
        let src_buffer_range = src_buffer_range.subrange(src_range);
        let dst_buffer_range = dst_buffer_range.subrange(dst_range);
        self.access(&[
            Access {
                range: src_buffer_range.buffer().gpu_address_subrange(src_buffer_range.range()),
                flags: AccessFlags::copy_read(),
            },
            Access {
                range: dst_buffer_range.buffer().gpu_address_subrange(dst_buffer_range.range()),
                flags: AccessFlags::copy_write(),
            },
        ]);
        self.command_buffer.encode_copy(src_buffer_range, dst_buffer_range);
    }

    pub fn encode_fill<Dst: AsBufferRangeMut<Buffer: Buffer<Backend = B>>>(
        &mut self,
        dst: &mut Dst,
        value: u8,
    ) {
        let dst_buffer_range = dst.as_buffer_range_mut();
        assert!(!dst_buffer_range.range().is_empty(), "zero-sized fills are not allowed");
        self.access(&[Access {
            range: dst_buffer_range.buffer().gpu_address_subrange(dst_buffer_range.range()),
            flags: AccessFlags::copy_write(),
        }]);
        self.command_buffer.encode_fill(dst_buffer_range, value);
    }

    pub fn access(
        &mut self,
        accesses: &[Access],
    ) {
        if let Some((after, before)) = self.hazard_tracker.access(accesses) {
            self.command_buffer.encode_barrier(after, before);
        }
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

    pub fn context(&self) -> &'encoding B::Context {
        self.context
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
    allocation_pool: Rc<AllocationPool<B>>,
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
    allocation_pool: Rc<AllocationPool<B>>,
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
    allocation_pool: Rc<AllocationPool<B>>,
}

impl<B: Backend> Completed<B> {
    pub fn gpu_execution_time(&self) -> Duration {
        self.command_buffer.gpu_execution_time()
    }
}
