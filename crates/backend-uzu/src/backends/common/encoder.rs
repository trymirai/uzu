use std::{
    mem::size_of_val,
    ops::{Bound, Range, RangeBounds},
    sync::Arc,
    time::Duration,
};

use bytemuck::{AnyBitPattern, NoUninit};

use crate::backends::common::{
    AccessFlags, Allocation, AllocationPool, AllocationType, AsBufferRangeMut, AsBufferRangeRef, Backend, Buffer,
    BufferGpuAddressRangeExt, CommandBuffer, CommandBufferCompleted, CommandBufferEncoding, CommandBufferExecutable,
    CommandBufferInitial, CommandBufferPending, Context,
    hazard_tracker::{Access, HazardTracker},
};
#[cfg(feature = "trace")]
use crate::{array::size_for_shape, data_type::DataType, trace::Recorder};

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
    allocation_pool: Arc<AllocationPool<B>>,
    hazard_tracker: HazardTracker,
    #[cfg(feature = "trace")]
    recorder: Option<Recorder<B>>,
}

impl<'encoding, B: Backend> Encoder<'encoding, B> {
    pub fn new(context: &'encoding B::Context) -> Result<Self, B::Error> {
        Self::new_with_name(context, None)
    }

    pub fn new_with_name(
        context: &'encoding B::Context,
        name: Option<&str>,
    ) -> Result<Self, B::Error> {
        Self::new_with_pool_name(context, Arc::new(context.create_allocation_pool(false)), name)
    }

    pub fn new_with_pool_name(
        context: &'encoding B::Context,
        allocation_pool: Arc<AllocationPool<B>>,
        name: Option<&str>,
    ) -> Result<Self, B::Error> {
        let command_buffer = context.create_command_buffer(name)?.start_encoding();
        let hazard_tracker = HazardTracker::new();

        Ok(Self {
            context,
            command_buffer,
            allocation_pool,
            hazard_tracker,
            #[cfg(feature = "trace")]
            recorder: None,
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

    pub fn allocate_constant_from_slice<T: NoUninit + AnyBitPattern>(
        &mut self,
        data: &[T],
    ) -> Result<Allocation<B>, B::Error> {
        let mut allocation = self.allocate_constant(size_of_val(data))?;
        allocation.copyin(data);
        Ok(allocation)
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

    pub fn push_debug_group(
        &mut self,
        name: &str,
    ) {
        self.command_buffer.push_debug_group(name);
    }

    pub fn pop_debug_group(&mut self) {
        self.command_buffer.pop_debug_group();
    }

    pub fn access(
        &mut self,
        accesses: &[Access],
    ) {
        if let Some((after, before)) = self.hazard_tracker.access(accesses) {
            self.command_buffer.encode_barrier(after, before);
        }
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
            #[cfg(feature = "trace")]
            recorder: self.recorder,
        }
    }
}

#[cfg(feature = "trace")]
impl<'encoding, B: Backend> Encoder<'encoding, B> {
    pub fn attach_recorder(
        &mut self,
        recorder: Recorder<B>,
    ) {
        self.recorder = Some(recorder);
    }

    pub fn is_recording(&self) -> bool {
        self.recorder.is_some()
    }

    pub fn push_trace_scope(
        &mut self,
        segment: std::fmt::Arguments<'_>,
    ) {
        if let Some(recorder) = &mut self.recorder {
            recorder.push_scope(segment);
        }
    }

    pub fn pop_trace_scope(&mut self) {
        if let Some(recorder) = &mut self.recorder {
            recorder.pop_scope();
        }
    }

    // Copies because encode-chain allocations are moved and their ranges recycled.
    // Destinations are global: the recorder outlives the encoder's pool.
    pub fn trace(
        &mut self,
        name: &str,
        src: &Allocation<B>,
        shape: &[usize],
        data_type: DataType,
    ) {
        let Some(mut recorder) = self.recorder.take() else {
            return;
        };

        let path = recorder.path(name);
        let byte_count = size_for_shape(shape, data_type);
        assert!(
            src.size() >= byte_count,
            "trace {path} declares {byte_count} bytes but the source allocation holds {}",
            src.size(),
        );

        let mut destination = self
            .context
            .create_allocation(byte_count, AllocationType::Global)
            .unwrap_or_else(|error| panic!("failed to allocate trace destination for {path}: {error:?}"));
        self.encode_copy(src, ..byte_count, &mut destination, ..);
        recorder.record(path, shape.into(), data_type, destination).expect("failed to record trace array");

        self.recorder = Some(recorder);
    }

    pub fn trace_host<T: NoUninit + AnyBitPattern>(
        &mut self,
        name: &str,
        data: &[T],
        shape: &[usize],
        data_type: DataType,
    ) {
        let Some(mut recorder) = self.recorder.take() else {
            return;
        };

        let path = recorder.path(name);
        let byte_count = size_for_shape(shape, data_type);
        assert_eq!(byte_count, size_of_val(data), "trace {path} declares a shape that does not match the data");

        let mut destination = self
            .context
            .create_allocation(byte_count, AllocationType::Global)
            .unwrap_or_else(|error| panic!("failed to allocate trace destination for {path}: {error:?}"));
        destination.copyin(data);
        recorder.record(path, shape.into(), data_type, destination).expect("failed to record trace array");

        self.recorder = Some(recorder);
    }
}

pub struct Executable<B: Backend> {
    command_buffer: <B::CommandBuffer as CommandBuffer>::Executable,
    allocation_pool: Arc<AllocationPool<B>>,
    #[cfg(feature = "trace")]
    recorder: Option<Recorder<B>>,
}

impl<B: Backend> Executable<B> {
    pub fn submit(self) -> Pending<B> {
        Pending {
            command_buffer: self.command_buffer.submit(),
            allocation_pool: self.allocation_pool,
            #[cfg(feature = "trace")]
            recorder: self.recorder,
        }
    }
}

pub struct Pending<B: Backend> {
    command_buffer: <B::CommandBuffer as CommandBuffer>::Pending,
    allocation_pool: Arc<AllocationPool<B>>,
    #[cfg(feature = "trace")]
    recorder: Option<Recorder<B>>,
}

impl<B: Backend> Pending<B> {
    pub fn wait_until_completed(self) -> Result<Completed<B>, B::Error> {
        Ok(Completed {
            command_buffer: self.command_buffer.wait_until_completed()?,
            _allocation_pool: self.allocation_pool,
            #[cfg(feature = "trace")]
            recorder: self.recorder,
        })
    }
}

pub struct Completed<B: Backend> {
    command_buffer: <B::CommandBuffer as CommandBuffer>::Completed,
    _allocation_pool: Arc<AllocationPool<B>>,
    #[cfg(feature = "trace")]
    recorder: Option<Recorder<B>>,
}

impl<B: Backend> Completed<B> {
    pub fn gpu_execution_time(&self) -> Duration {
        self.command_buffer.gpu_execution_time()
    }

    #[cfg(feature = "trace")]
    pub fn take_recorder(&mut self) -> Option<Recorder<B>> {
        self.recorder.take()
    }
}
