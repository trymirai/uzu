use std::{
    cell::RefCell,
    ops::Range,
    pin::Pin,
    rc::{Rc, Weak},
};

use super::{RangeAllocationType, RangeAllocator};
use crate::{
    ArrayElement,
    backends::common::{
        Backend, Buffer, Context, DenseBuffer,
        buffer_range::{AsBufferRangeMut, AsBufferRangeRef, BufferRangeMut, BufferRangeRef},
    },
};

pub struct Allocation<B: Backend> {
    allocator: Rc<Allocator<B>>,
    buffer: *const B::DenseBuffer,
    range: Range<usize>,
    allocation_type: RangeAllocationType,
}

impl<B: Backend> Allocation<B> {
    pub fn copyout<T: ArrayElement>(&self) -> Vec<T> {
        let buffer_range = self.as_buffer_range_ref();
        let (buffer, range) = (buffer_range.buffer(), buffer_range.range());
        let bytes = unsafe {
            std::slice::from_raw_parts((buffer.cpu_ptr().as_ptr() as *const u8).add(range.start), range.len())
        };
        bytemuck::cast_slice(bytes).to_vec()
    }
}

impl<B: Backend> AsBufferRangeRef for Allocation<B> {
    type Buffer = B::DenseBuffer;

    fn as_buffer_range_ref<'a>(&'a self) -> BufferRangeRef<'a, B::DenseBuffer> {
        // SAFETY: we keep a strong ref to the allocator that owns the bufs and won't deallocate them while we're alive
        let buffer = unsafe { &*self.buffer };

        BufferRangeRef::new(buffer, self.range.clone())
    }
}

impl<B: Backend> AsBufferRangeMut for Allocation<B> {
    fn as_buffer_range_mut<'a>(&'a mut self) -> BufferRangeMut<'a, B::DenseBuffer> {
        // SAFETY: we keep a strong ref to the allocator that owns the bufs and won't deallocate them while we're alive
        let buffer = unsafe { &*self.buffer };
        // SAFETY: allocator algorithm (hopefully if there is no bugs) guarantees no two overlapping live allocations can exist (which is the contract of BufferRangeMut)
        unsafe { BufferRangeMut::new_shared(buffer, self.range.clone()) }
    }
}

impl<B: Backend> Drop for Allocation<B> {
    fn drop(&mut self) {
        self.allocator.free(self)
    }
}

pub struct AllocationPool<B: Backend> {
    reusable: bool,
    allocator: Rc<Allocator<B>>,
    pool_number: usize,
}

impl<B: Backend> Drop for AllocationPool<B> {
    fn drop(&mut self) {
        self.allocator.free_pool(self)
    }
}

pub enum AllocationType<'a, B: Backend> {
    Global,
    Pooled {
        pool: &'a AllocationPool<B>,
        cpu_available: bool,
    },
}

struct AllocatorBuffer<B: Backend> {
    buffer: Pin<Box<B::DenseBuffer>>,
    range_allocator: RangeAllocator,
}

pub struct Allocator<B: Backend> {
    context: Weak<B::Context>,
    allocator_buffers: RefCell<Vec<AllocatorBuffer<B>>>,
    next_pool_number: RefCell<usize>,
    peak_memory_usage: RefCell<usize>,
}

impl<B: Backend> Allocator<B> {
    pub fn new(context: Weak<B::Context>) -> Rc<Self> {
        Rc::new(Self {
            context,
            allocator_buffers: RefCell::new(Vec::new()),
            next_pool_number: RefCell::new(0),
            peak_memory_usage: RefCell::new(0),
        })
    }

    pub fn allocate(
        self: &Rc<Self>,
        size: usize,
        allocation_type: AllocationType<B>,
    ) -> Result<Allocation<B>, B::Error> {
        assert!(size > 0, "allocation size must be greater than 0");
        let alignment = usize::clamp(size.next_power_of_two(), B::MIN_ALLOCATION_ALIGNMENT, 16_384);
        let allocation_type = match allocation_type {
            AllocationType::Global => RangeAllocationType::Global,
            AllocationType::Pooled {
                pool,
                cpu_available,
            } => RangeAllocationType::Pooled {
                pool: pool.pool_number,
                can_alias_before: !cpu_available,
                can_alias_after: !(cpu_available && pool.reusable),
            },
        };

        let mut allocator_buffers = self.allocator_buffers.borrow_mut();

        let found = allocator_buffers.iter_mut().enumerate().find_map(|(allocator_buffer_index, allocator_buffer)| {
            let range = allocator_buffer.range_allocator.allocate_range_aligned(size, alignment, allocation_type)?;
            let buffer = allocator_buffer.buffer.as_ref().get_ref() as *const B::DenseBuffer;

            Some((allocator_buffer_index, buffer, range))
        });

        let (buffer, range) = if let Some((allocator_buffer_index, buffer, range)) = found {
            Self::restore_buffer_order(&mut allocator_buffers, allocator_buffer_index);

            (buffer, range)
        } else {
            let new_allocator_buffer_size = usize::max(size, 268_435_456);

            let mut allocator_buffer = AllocatorBuffer::<B> {
                buffer: Box::pin(self.context.upgrade().unwrap().create_buffer(new_allocator_buffer_size)?), // Upgrade can never fail
                range_allocator: RangeAllocator::new(0..new_allocator_buffer_size),
            };

            let buffer = allocator_buffer.buffer.as_ref().get_ref() as *const B::DenseBuffer;
            let range =
                allocator_buffer.range_allocator.allocate_range_aligned(size, alignment, allocation_type).unwrap(); // Can never fail

            allocator_buffers.push(allocator_buffer);
            let allocator_buffer_index = allocator_buffers.len() - 1;
            Self::restore_buffer_order(&mut allocator_buffers, allocator_buffer_index);

            *self.peak_memory_usage.borrow_mut() =
                allocator_buffers.iter().map(|allocator_buffer| allocator_buffer.buffer.size()).sum();

            (buffer, range)
        };

        Ok(Allocation {
            allocator: self.clone(),
            buffer,
            range,
            allocation_type,
        })
    }

    pub fn create_pool(
        self: &Rc<Self>,
        reusable: bool,
    ) -> AllocationPool<B> {
        let pool_number = *self.next_pool_number.borrow();

        *self.next_pool_number.borrow_mut() += 1;

        AllocationPool {
            reusable,
            allocator: self.clone(),
            pool_number,
        }
    }

    pub fn peak_memory_usage(self: Rc<Self>) -> usize {
        *self.peak_memory_usage.borrow()
    }

    // TODO: Maybe hysteresis in free/free_pool?

    fn free(
        self: &Rc<Self>,
        allocation: &Allocation<B>,
    ) {
        let mut allocator_buffers = self.allocator_buffers.borrow_mut();

        let allocator_buffer_index = allocator_buffers
            .iter()
            .position(|allocator_buffer| {
                (allocator_buffer.buffer.as_ref().get_ref() as *const B::DenseBuffer) == allocation.buffer
            })
            .unwrap(); // Can never fail

        allocator_buffers[allocator_buffer_index]
            .range_allocator
            .free_range(allocation.range.clone(), allocation.allocation_type);

        if allocator_buffers[allocator_buffer_index].range_allocator.is_empty() {
            allocator_buffers.remove(allocator_buffer_index);
        } else {
            Self::restore_buffer_order(&mut allocator_buffers, allocator_buffer_index);
        }
    }

    fn free_pool(
        self: &Rc<Self>,
        pool: &AllocationPool<B>,
    ) {
        let mut allocator_buffers = self.allocator_buffers.borrow_mut();

        allocator_buffers.retain_mut(|allocation_buffer| {
            allocation_buffer.range_allocator.free_pool(pool.pool_number);
            !allocation_buffer.range_allocator.is_empty()
        });

        if allocator_buffers.len() > 1 {
            allocator_buffers.sort_by_key(|allocator_buffer| allocator_buffer.range_allocator.total_available());
        }
    }

    fn restore_buffer_order(
        allocator_buffers: &mut [AllocatorBuffer<B>],
        mut index: usize,
    ) {
        while index > 0
            && allocator_buffers[index].range_allocator.total_available()
                < allocator_buffers[index - 1].range_allocator.total_available()
        {
            allocator_buffers.swap(index, index - 1);
            index -= 1;
        }

        while index + 1 < allocator_buffers.len()
            && allocator_buffers[index].range_allocator.total_available()
                > allocator_buffers[index + 1].range_allocator.total_available()
        {
            allocator_buffers.swap(index, index + 1);
            index += 1;
        }
    }
}

#[cfg(all(test, metal_backend))]
#[path = "../../../../tests/unit/backends/common/allocator/allocator.rs"]
mod tests;
